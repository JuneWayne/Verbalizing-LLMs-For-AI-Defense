import { tutorialBounds } from './tutorial-bounds.js?v=tour-fit-31';
import { modelSummary } from './model-copy.js?v=plain-math-29';
import * as THREE from 'three';
import { theme } from './theme.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { advanceSpring, nearestSlot } from './carousel-motion.js';
import { createBlockHover } from './block-hover.js?v=selected-value-17';

const ui = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
// Keep the architecture legend below the title as it wraps on smaller screens.
const titleObserver = new ResizeObserver(() => {
  document.querySelector('.legend').style.top = (document.querySelector('header').getBoundingClientRect().bottom + 16) + 'px';
});
titleObserver.observe(document.querySelector('header'));
const response = await fetch('./models.json?v=plain-math-29');
if (!response.ok) throw new Error('Model metadata unavailable');
const models = await response.json();
await document.fonts.load('20px "Viewer Comic"');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const wrap = value => ((value % models.length) + models.length) % models.length;
let selected = Math.max(0, models.findIndex(model => model.id === (new URLSearchParams(location.search).get('model') || 'antares-1b')));
let position = selected, target = selected, velocity = 0, paused = reduced.matches;
let renderedSelection = -1, needsRender = true, previousHover = -1, hoverChangedAt = 0;
let previousTime = performance.now(), orbitTime = 0, hovered = -1, gesture = null, wheelTimer;

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-20, 20, 12, -12, .1, 200);
camera.position.set(0, 17, 36); camera.lookAt(0, 0, 0);
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setClearColor(0, 0);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
ui.architecture.append(renderer.domElement);
const pmrem = new THREE.PMREMGenerator(renderer), room = new RoomEnvironment();
scene.environment = pmrem.fromScene(room, .06).texture;
scene.environmentIntensity = .55;
room.dispose(); pmrem.dispose();
scene.add(new THREE.HemisphereLight('#ffffff', '#bcc9dd', 1.5));
const light = new THREE.DirectionalLight('#ffffff', 2.3);
light.position.set(-10, 18, 16); scene.add(light);

// Shared geometry and instanced parts keep all seven models loaded affordably.
const box = new RoundedBoxGeometry(1, 1, 1, 1, .065);
const matrix = new THREE.Matrix4(), rotation = new THREE.Quaternion();
function instances(parent, entries, color, opacity = 1) {
  const material = new THREE.MeshStandardMaterial({color, roughness: .3, metalness: .12, transparent: opacity < 1, opacity, depthWrite: opacity === 1});
  const mesh = new THREE.InstancedMesh(box, material, entries.length);
  entries.forEach(([x, y, z, w, h, d], i) => {
    matrix.compose(new THREE.Vector3(x, y, z), rotation, new THREE.Vector3(w, h, d));
    mesh.setMatrixAt(i, matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  parent.add(mesh);
  return mesh;
}
const shadowCanvas = document.createElement('canvas');
shadowCanvas.width = shadowCanvas.height = 128;
const context = shadowCanvas.getContext('2d');
const gradient = context.createRadialGradient(64, 64, 3, 64, 64, 64);
gradient.addColorStop(0, 'rgba(35,45,75,.28)'); gradient.addColorStop(1, 'rgba(35,45,75,0)');
context.fillStyle = gradient; context.fillRect(0, 0, 128, 128);
const shadowTexture = new THREE.CanvasTexture(shadowCanvas);
const hitTargets = [];

// Every shell is one actual decoder layer. Widths use a compressed scale.
function buildModel(model, index) {
  const group = new THREE.Group(); scene.add(group);
  const height = 2.7 + Math.log2(model.hidden / 896) * .55;
  const mlpDepth = 1.1 + Math.log2(model.mlp / 2048) * .23;
  const spacing = .51, length = model.layers * spacing + .65, depth = 2.7 + mlpDepth;
  const shells = [], norms = [], attention = [], fullAttention = [], convolution = [], mlp = [], output = [], rails = [], heads = [];
  const layers = [];
  model.layerTypes.forEach((type, i) => {
    const x = (i - (model.layers - 1) / 2) * spacing;
    layers.push({index: i, type, hidden: model.hidden, mlp: model.mlp});
    shells.push([x, 0, 0, .44, height + .5, depth]);
    norms.push([x, height * .48, 0, .31, .09, depth - .3], [x, -.1, 0, .31, .09, depth - .3]);
    (type === 'full_attention' ? fullAttention : attention).push([x, height * .25, 0, .31, height * (type === 'linear_attention' ? .27 : .38), depth * .72]);
    if (type === 'linear_attention') {
      // The four small blocks represent this model's four-position causal convolution.
      for (let tap = 0; tap < model.linearConvKernel; tap++) convolution.push([x, height * .055, (tap - (model.linearConvKernel - 1) / 2) * .65, .31, .15, .5]);
    }
    // Two gated MLP branches and their output projection, shared by these families.
    mlp.push([x, -height * .25, .55, .31, height * .37, mlpDepth * .6], [x, -height * .25, -.55, .31, height * .37, mlpDepth * .6]);
    output.push([x, -height * .48, 0, .31, .12, depth * .72]);
    for (const z of [-depth / 2, depth / 2]) rails.push([x, 0, z, .06, height + .5, .06]);
    rails.push([x, (height + .5) / 2, 0, .06, .06, depth]);
    const count = type === 'linear_attention' ? model.linearValueHeads : model.heads;
    // Show configured head divisions on the end layers; interior copies are too small to read.
    for (let head = 0; (i === 0 || i === model.layers - 1) && head < count; head++) heads.push([x, height * .43, -depth * .3 + (head + .5) / count * depth * .6, .3, .1, .65 / count]);
  });
  const colored = [
    instances(group, norms, '#d6e1ef'), instances(group, attention, '#628865'),
    instances(group, fullAttention, '#8f91bc'), instances(group, mlp, '#5b90b9'),
    instances(group, output, '#6791b0'), instances(group, heads, '#a0b492'),
    instances(group, rails, '#b7c8dc'), instances(group, convolution, '#8cb2a2')
  ];
  colored.forEach(mesh => {mesh.userData.color = mesh.material.color.clone();});
  instances(group, shells, '#f1f6ff', .045);
  instances(group, [[-length / 2, 0, 0, .35, height + .6, depth], [length / 2, 0, 0, .35, height + .6, depth]], '#c5d3e3');
  const hit = new THREE.Mesh(new THREE.BoxGeometry(length + .4, height + .6, depth), new THREE.MeshBasicMaterial({visible: false}));
  hit.userData.index = index; group.add(hit); hitTargets.push(hit);
  const highlight = createBlockHover(group, new THREE.Vector3(length + .5, height + .7, depth + .1));
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(length * 1.18, depth * 2.1), new THREE.MeshBasicMaterial({map: shadowTexture, transparent: true, depthWrite: false}));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = -height / 2 - .55; group.add(shadow);
  const label = document.createElement('button'); label.className = 'orbit-label'; label.textContent = model.name;
  label.setAttribute('aria-label', 'Select ' + model.name);
  label.onclick = () => selectModel(index);
  ui['model-labels'].append(label);
  return {group, layers, colored, height, length, depth, label, highlight};
}
const assemblies = models.map(buildModel);

function textSprite(text) {
  const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d'); ctx.font = '28px "Viewer Comic"'; ctx.fillStyle = theme.title; ctx.textAlign = 'center'; ctx.fillText(text, 128, 43);
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  const object = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, opacity: .48, depthWrite: false}));
  object.scale.set(2.2, .55, 1); scene.add(object); return object;
}
const tokens = Array.from({length: 32}, (_, i) => textSprite(['0.142', 'safe', '0.031', 'injection', '2048', 'prompt', '0.84', 'token'][i % 8]));
const ringPoints = Array.from({length: 161}, (_, i) => new THREE.Vector3(Math.sin(i / 160 * Math.PI * 2) * 24, -2.8, Math.cos(i / 160 * Math.PI * 2) * 13 - 4));
scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPoints), new THREE.LineBasicMaterial({color: '#c0cce0', transparent: true, opacity: .65})));

function updateSelection() {
  selected = wrap(Math.round(target));
  if (renderedSelection === selected) return;
  renderedSelection = selected; hovered = -1; needsRender = true;
  const model = models[selected];
  ui['model-title'].textContent = ui['model-picker'].textContent = model.name;
  const summary = modelSummary(model);
  ui['model-size'].textContent = summary.size;
  ui['model-detail'].textContent = summary.detail;
  ui['model-detail'].title = summary.architecture + ' ' + summary.attention;
  document.querySelectorAll('.model-option').forEach((button, i) => button.setAttribute('aria-current', String(i === selected)));
  history.replaceState(null, '', '?model=' + model.id);
}
function selectModel(index) {
  clearTimeout(wheelTimer);
  target = nearestSlot(wrap(index), target, models.length);
  updateSelection();
}
function stepModel(step) { clearTimeout(wheelTimer); target = Math.round(target) + step; updateSelection(); }
function openModel() {
  location.href = './viewer.html?model=' + encodeURIComponent(models[selected].id);
}
ui['previous-model'].onclick = () => stepModel(-1);
ui['next-model'].onclick = () => stepModel(1);
ui['choose-model'].onclick = ui['model-picker'].onclick = () => ui['model-dialog'].showModal();
ui['close-models'].onclick = () => ui['model-dialog'].close();
ui['model-dialog'].addEventListener('click', event => {
  const bounds = ui['model-dialog'].getBoundingClientRect();
  if (event.target === ui['model-dialog'] && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) ui['model-dialog'].close();
});
models.forEach((model, index) => {
  const button = document.createElement('button'); button.className = 'model-option';
  const title = document.createElement('span'); title.textContent = model.name;
  const detail = document.createElement('small'); detail.textContent = model.layers + ' layers · ' + (model.viewer ? 'Recorded prompts' : 'Architecture preview');
  button.append(title, detail); button.onclick = () => {selectModel(index); ui['model-dialog'].close();};
  ui['model-list'].append(button);
});

const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
function pick(event) {
  pointer.set(event.clientX / innerWidth * 2 - 1, 1 - event.clientY / innerHeight * 2);
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(hitTargets, false)[0]?.object.userData.index ?? -1;
}
ui.architecture.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  clearTimeout(wheelTimer);
  gesture = {id: event.pointerId, x: event.clientX, y: event.clientY, start: target, hit: pick(event), dragged: false};
  ui.architecture.setPointerCapture(event.pointerId);
});
ui.architecture.addEventListener('pointermove', event => {
  if (gesture) {
    const dx = event.clientX - gesture.x;
    if (Math.abs(dx) > 8) gesture.dragged = true;
    if (gesture.dragged) {target = gesture.start - dx / Math.min(320, innerWidth * .65); updateSelection();}
  } else hovered = pick(event);
  renderer.domElement.style.cursor = gesture?.dragged ? 'grabbing' : hovered >= 0 ? 'pointer' : 'grab';
});
ui.architecture.addEventListener('pointerup', event => {
  if (!gesture) return;
  const clickedModel = gesture.hit;
  const dragged = gesture.dragged || Math.abs(event.clientY - gesture.y) > 8; gesture = null;
  ui.architecture.releasePointerCapture(event.pointerId);
  if (dragged) {target = Math.round(target); updateSelection(); return;}
  const index = clickedModel;
  if (index === selected) openModel(); else if (index >= 0) selectModel(index);
});
ui.architecture.addEventListener('pointercancel', () => {gesture = null; target = Math.round(target); updateSelection();});
ui.architecture.addEventListener('pointerleave', () => {hovered = -1;});
ui.architecture.addEventListener('wheel', event => {
  if (event.ctrlKey) return;
  event.preventDefault();
  const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
  const pixels = delta * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
  target += THREE.MathUtils.clamp(pixels / 260, -.7, .7);
  updateSelection();
  clearTimeout(wheelTimer);
  wheelTimer = setTimeout(() => {target = Math.round(target); updateSelection();}, 160);
}, {passive: false});
ui.architecture.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {event.preventDefault(); stepModel(event.key === 'ArrowRight' ? 1 : -1);}
  if (event.key === 'Enter') {event.preventDefault(); openModel();}
});
reduced.addEventListener('change', () => {paused = reduced.matches; needsRender = true;});

function resize() {
  needsRender = true;
  const halfWidth = innerWidth < 650 ? 14 : 19.5;
  const halfHeight = Math.max(9.5, halfWidth * innerHeight / innerWidth);
  const width = halfHeight * innerWidth / innerHeight;
  Object.assign(camera, {left: -width, right: width, top: halfHeight, bottom: -halfHeight});
  camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);
}
addEventListener('resize', resize); resize(); updateSelection();
const grey = new THREE.Color('#bdc6d4'), projected = new THREE.Vector3();
function animate(now) {
  requestAnimationFrame(animate);
  const seconds = Math.min((now - previousTime) / 1000, .1); previousTime = now;
  if (document.hidden) return;
  if (reduced.matches) {position = target; velocity = 0;}
  else ({position, velocity} = advanceSpring(position, velocity, target, seconds));
  if (hovered !== previousHover) {previousHover = hovered; hoverChangedAt = now; needsRender = true;}
  const settled = Math.abs(position - target) < .0001 && Math.abs(velocity) < .0001;
  if ((paused || reduced.matches) && settled && !needsRender && now - hoverChangedAt > 250) return;
  needsRender = false;
  if (!paused && !reduced.matches) orbitTime += seconds;
  assemblies.forEach((model, index) => {
    const angle = (index - position) * Math.PI * 2 / models.length;
    const front = (Math.cos(angle) + 1) / 2, emphasis = Math.pow(front, 5);
    model.group.position.set(Math.sin(angle) * 20, .3 + emphasis * (innerHeight < 650 && innerWidth > 650 ? 6.3 : 1.5), Math.cos(angle) * 12 - 5);
    model.group.rotation.y = .12 - Math.sin(angle) * .72;
    model.group.scale.setScalar(.32 + emphasis * .88);
    model.colored.forEach(mesh => mesh.material.color.copy(mesh.userData.color).lerp(grey, (1 - emphasis) * .85));
    model.highlight.set(index === hovered);
    model.highlight.animate(seconds, reduced.matches);
    projected.set(0, -model.height / 2 - 1, 0); model.group.localToWorld(projected); projected.project(camera);
    const x = (projected.x + 1) * innerWidth / 2, y = (1 - projected.y) * innerHeight / 2;
    model.label.style.transform = 'translate(-50%, -50%) translate(' + x + 'px,' + y + 'px)';
    model.label.style.opacity = String(.4 + front * .6);
    model.label.hidden = emphasis > .85 || x < 80 || x > innerWidth - 80 || y < 100 || y > innerHeight - 210;
  });
  tokens.forEach((token, i) => {
    const angle = i / tokens.length * Math.PI * 2 + orbitTime * .07;
    token.position.set(Math.sin(angle) * 25, -2.8 + Math.sin(angle * 2) * 1.1, Math.cos(angle) * 14 - 4);
    projected.copy(token.position).project(camera);
    // Keep decorative text outside the selected model's name and controls.
    token.material.opacity = projected.y < -.43 && Math.abs(projected.x) < .45 ? 0 : .48;
  });
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// Diagnostics expose persistent scene identities for animation and navigation tests.
window.modelLanding = {models, selectModel, assemblies, renderer, camera, get selected() {return selected;}, get position() {return position;}, get target() {return target;}, get layers() {return assemblies[selected].layers;}};

ui.architecture.tutorialBounds = () => tutorialBounds([assemblies[selected].group], camera, renderer.domElement);
