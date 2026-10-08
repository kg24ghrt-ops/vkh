const REGIONS = {
  us: { w: 215.9, h: 279.4, name: 'Letter', color: [1.0, 0.995, 0.97], grain: 1.0 },
  eu: { w: 210, h: 297, name: 'A4', color: [1.0, 0.99, 0.97], grain: 1.1 },
  asia: { w: 182, h: 257, name: 'B5', color: [0.99, 0.995, 1.0], grain: 0.9 },
  latam: { w: 215.9, h: 330, name: 'Oficio', color: [1.0, 0.985, 0.96], grain: 1.05 }
};

const WGSL = `
struct Uniforms {
  aspect: f32,
  lightDir: vec3f,
  color: vec3f,
  grain: f32,
  time: f32,
  curl: f32,
  _pad: vec2f,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

fn hash(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1, 0)), u.x),
             mix(hash(i + vec2f(0, 1)), hash(i + vec2f(1, 1)), u.x), u.y);
}

fn fbm(p: vec2f, octaves: i32) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var freq = 1.0;
  for (var i = 0; i < 5; i++) {
    if (i >= octaves) break;
    sum += noise(p * freq) * amp;
    amp *= 0.5;
    freq *= 2.0;
  }
  return sum;
}

fn normalMap(uv: vec2f, scale: f32) -> vec3f {
  let eps = 1.0 / 512.0;
  let h = fbm(uv * scale, 4);
  let hx = fbm(uv * scale + vec2f(eps, 0), 4);
  let hy = fbm(uv * scale + vec2f(0, eps), 4);
  let n = normalize(vec3f(hx - h, hy - h, eps * scale));
  return n;
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
  let pos = array<vec2f, 4>(
    vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, 1)
  );
  let uv = array<vec2f, 4>(
    vec2f(0, 1), vec2f(1, 1), vec2f(0, 0), vec2f(1, 0)
  );
  return vec4f(pos[idx], 0, 1);
}

@fragment
fn fs_main(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let uv = frag.xy / vec2f(1, u.aspect) * 0.5 + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec4f(0, 0, 0, 0);
  }

  let n = normalMap(uv, 80.0 * u.grain);
  let fiber = fbm(uv * 200.0, 3) * 0.02;
  let grain = fbm(uv * 500.0, 2) * 0.015;

  let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  let edgeShade = smoothstep(0.0, 0.03, edge) * 0.15;

  let curl = u.curl * smoothstep(0.0, 0.15, min(uv.x, 1.0 - uv.x)) *
             smoothstep(0.0, 0.15, min(uv.y, 1.0 - uv.y));

  let light = normalize(u.lightDir);
  let diffuse = max(dot(n, light), 0.0) * 0.4 + 0.6;
  let specular = pow(max(dot(reflect(-light, n), vec3f(0, 0, 1)), 0.0), 64.0) * 0.3;

  let base = u.color * (1.0 + fiber + grain - edgeShade - curl);
  let col = base * diffuse + specular;

  let shadow = smoothstep(0.95, 1.0, max(uv.x, uv.y)) * 0.2;
  return vec4f(col * (1.0 - shadow), 1.0);
}
`;

let device = null, context = null, pipeline = null, bindGroupLayout = null;
let uniformBuffer = null, uniformData = null;
let currentRegion = 'us';
let lightDir = [0.3, 0.3, 1.0];
let animationId = null;
let needsRender = true;
let reducedMotion = false;

const canvases = {};
const panels = {};
const tabs = {};

async function initWebGPU() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    device = await adapter.requestDevice({
      requiredLimits: { maxUniformBufferBindingSize: 256 }
    });
    device.lost.then(() => { device = null; initFallback(); });
    return true;
  } catch { return false; }
}

function setupCanvas(canvas) {
  if (!device) return;
  context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const module = device.createShaderModule({ code: WGSL });
  bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }]
  });
  pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-strip' }
  });

  uniformBuffer = device.createBuffer({
    size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  uniformData = new Float32Array(24);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
  });
  return { context, bindGroup };
}

function render(region) {
  if (!device || !canvases[region]) return;
  const canvas = canvases[region];
  const { context, bindGroup } = canvas.gpuData;
  const r = REGIONS[region];
  const dpr = Math.min(window.devicePixelRatio, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);

  const aspect = r.w / r.h;
  uniformData[0] = aspect;
  uniformData[1] = lightDir[0];
  uniformData[2] = lightDir[1];
  uniformData[3] = lightDir[2];
  uniformData[4] = r.color[0];
  uniformData[5] = r.color[1];
  uniformData[6] = r.color[2];
  uniformData[7] = r.grain;
  uniformData[8] = performance.now() / 1000;
  uniformData[9] = reducedMotion ? 0 : 0.08;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: 'clear', storeOp: 'store'
    }]
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(4);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function renderAll() {
  Object.keys(canvases).forEach(r => {
    if (panels[r].classList.contains('active')) render(r);
  });
  needsRender = false;
}

function scheduleRender() {
  if (needsRender) return;
  needsRender = true;
  if (document.visibilityState === 'visible') requestAnimationFrame(renderAll);
}

function initFallback() {
  document.getElementById('gpu-notice').classList.remove('hidden');
  document.querySelectorAll('.sheet-canvas').forEach(c => {
    c.style.display = 'none';
    const panel = c.closest('.sheet-panel');
    const r = c.dataset.region;
    const reg = REGIONS[r];
    panel.style.aspectRatio = `${reg.w} / ${reg.h}`;
    panel.style.background = `linear-gradient(135deg, 
      rgb(${reg.color[0]*255},${reg.color[1]*255},${reg.color[2]*255}) 0%,
      rgb(${Math.round(reg.color[0]*245)},${Math.round(reg.color[1]*245)},${Math.round(reg.color[2]*245)}) 100%)`;
    panel.style.backgroundSize = '200% 200%';
    panel.style.borderRadius = '4px';
    panel.style.boxShadow = '0 10px 30px var(--sheet-shadow), 0 2px 8px var(--sheet-shadow)';
    panel.style.position = 'relative';
    panel.style.overflow = 'hidden';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none';
    svg.innerHTML = `
      <defs>
        <filter id="paper-${r}">
          <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="4" stitchTiles="stitch" result="noise"/>
          <feColorMatrix in="noise" type="saturate" values="0"/>
          <feBlend in="SourceGraphic" mode="multiply"/>
        </filter>
      </defs>
      <rect width="100%" height="100%" filter="url(#paper-${r})" opacity="0.15"/>
    `;
    panel.appendChild(svg);
  });
}

function switchRegion(region) {
  currentRegion = region;
  Object.keys(tabs).forEach(r => {
    tabs[r].setAttribute('aria-selected', r === region);
    panels[r].classList.toggle('active', r === region);
    panels[r].classList.toggle('hidden', r !== region);
  });
  scheduleRender();
}

function onPointerMove(e) {
  if (reducedMotion) return;
  const canvas = canvases[currentRegion];
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width * 2 - 1;
  const y = (e.clientY - rect.top) / rect.height * 2 - 1;
  lightDir[0] = x * 0.5;
  lightDir[1] = -y * 0.5;
  lightDir[2] = 1.0;
  const len = Math.hypot(...lightDir);
  lightDir = lightDir.map(v => v / len);
  scheduleRender();
}

function onDeviceOrientation(e) {
  if (reducedMotion) return;
  const beta = e.beta || 0;
  const gamma = e.gamma || 0;
  lightDir[0] = gamma / 90 * 0.5;
  lightDir[1] = beta / 90 * 0.5;
  lightDir[2] = 1.0;
  const len = Math.hypot(...lightDir);
  lightDir = lightDir.map(v => v / len);
  scheduleRender();
}

function onResize() {
  scheduleRender();
}

function onVisibilityChange() {
  if (document.visibilityState === 'visible') scheduleRender();
}

function onTabClick(e) {
  const tab = e.target.closest('[role="tab"]');
  if (tab) switchRegion(tab.dataset.region);
}

function onTabKeydown(e) {
  const tab = e.target.closest('[role="tab"]');
  if (!tab) return;
  const tabsList = Array.from(document.querySelectorAll('[role="tab"]'));
  const idx = tabsList.indexOf(tab);
  let nextIdx = idx;
  if (e.key === 'ArrowRight') nextIdx = (idx + 1) % tabsList.length;
  else if (e.key === 'ArrowLeft') nextIdx = (idx - 1 + tabsList.length) % tabsList.length;
  else if (e.key === 'Home') nextIdx = 0;
  else if (e.key === 'End') nextIdx = tabsList.length - 1;
  else return;
  e.preventDefault();
  tabsList[nextIdx].focus();
  switchRegion(tabsList[nextIdx].dataset.region);
}

async function init() {
  reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  document.querySelectorAll('[role="tab"]').forEach(t => tabs[t.dataset.region] = t);
  document.querySelectorAll('.sheet-panel').forEach(p => panels[p.id.replace('panel-', '')] = p);
  document.querySelectorAll('.sheet-canvas').forEach(c => canvases[c.dataset.region] = c);

  document.querySelector('.tabs').addEventListener('click', onTabClick);
  document.querySelector('.tabs').addEventListener('keydown', onTabKeydown);
  window.addEventListener('resize', onResize);
  document.addEventListener('visibilitychange', onVisibilityChange);
  document.addEventListener('pointermove', onPointerMove);
  window.addEventListener('deviceorientation', onDeviceOrientation);

  const hasGPU = await initWebGPU();
  if (hasGPU) {
    document.querySelectorAll('.sheet-canvas').forEach(c => {
      c.gpuData = setupCanvas(c);
    });
    renderAll();
  } else {
    initFallback();
  }
}

init();