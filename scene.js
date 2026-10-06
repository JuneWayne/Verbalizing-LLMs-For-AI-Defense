import { tutorialBounds } from './tutorial-bounds.js?v=tour-fit-31';
import { blockTitles, formatProbability } from './block-labels.js?v=transformer-terms-34';
import * as THREE from 'three';
import { createBlockHover } from './block-hover.js?v=selected-value-17';
import { createFlowWire } from './flow-wire.js?v=value-flow-13';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

// The scene draws recorded values. Animation illustrates their path, not GPU timing.
export function createScene(container, onLayer, onNumber, layerCount = 39) {
  const scene = new THREE.Scene();
  // A transparent canvas keeps the same cream page background behind every view.
  const camera = new THREE.OrthographicCamera(-25, 25, 15, -15, .1, 180);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0, 0);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.toneMapping = THREE.NoToneMapping;
  container.append(renderer.domElement);
  const labels = new CSS2DRenderer();
  labels.domElement.style.cssText = 'position:absolute;inset:0;pointer-events:none';
  container.append(labels.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  controls.enableDamping = !reduced.matches;
  controls.minZoom = .55;
  controls.maxZoom = 4;
  controls.maxPolarAngle = Math.PI * .49;
  controls.listenToKeyEvents(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('aria-label', 'Interactive model');
  scene.add(new THREE.HemisphereLight('#ffffff', '#ccd5e1', 2.1));
  const light = new THREE.DirectionalLight('#ffffff', 2);
  light.position.set(-15, 24, 20);
  scene.add(light);

  const stacks = new THREE.Group(), inside = new THREE.Group();
  scene.add(stacks, inside);
  inside.visible = false;
  const plates = [], stages = [], interactive = [], flows = [];
  let stackWidth = 1, stackHeight = 1;
  let inspecting = false;
  let detail = false, hasData = false, playing = false, hovered = null, transition = null, pending = false, lastTime = performance.now(), currentFrame = null;
  let openedAt = 0, revealElapsed = Infinity, selected = 24, selectedKind = 'jlens';
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const shadowCanvas = document.createElement('canvas');
  shadowCanvas.width = shadowCanvas.height = 128;
  const shadowContext = shadowCanvas.getContext('2d');
  const gradient = shadowContext.createRadialGradient(64, 64, 5, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(35,51,72,.23)');
  gradient.addColorStop(1, 'rgba(35,51,72,0)');
  shadowContext.fillStyle = gradient;
  shadowContext.fillRect(0, 0, 128, 128);
  const shadowTexture = new THREE.CanvasTexture(shadowCanvas);

  function label(parent, text, x, y, z = 0, className = 'component-label') {
    const element = document.createElement(['component-label', 'stack-title', 'layer-word'].includes(className) ? 'button' : 'div');
    element.className = className;
    element.textContent = text;
    const object = new CSS2DObject(element);
    object.position.set(x, y, z);
    parent.add(object);
    element.object3D = object;
    return element;
  }
  function shadow(parent, x, y, z, width, depth) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), new THREE.MeshBasicMaterial({ map: shadowTexture, transparent: true, depthWrite: false, opacity: .65 }));
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x, y, z);
    parent.add(mesh);
    return mesh;
  }
  function line(parent, points, color = '#a9b7c9') {
    const curve = new THREE.CatmullRomCurve3(points.map(point => new THREE.Vector3(...point)), false, 'catmullrom', .05);
    const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40));
    parent.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .65 })));
    return curve;
  }
  function flow(points, method = null) {
    const group = new THREE.Group();
    group.userData.method = method;
    inside.add(group);
    const curve = new THREE.CatmullRomCurve3(points.map(point => new THREE.Vector3(...point)), false, 'catmullrom', .05);
    const wire = createFlowWire(curve);
    group.add(wire.group);
    flows.push({wire, method, group});
  }

  function color(value) {
    return new THREE.Color('#f2f3f5').lerp(new THREE.Color(value < 0 ? '#709dd7' : '#eb9587'), Math.min(Math.abs(value) / 20, 1));
  }
  const geometry = new RoundedBoxGeometry(10.4, .45, 6, 3, .08);
  for (const [kind, x, title] of [['jlens', -14, 'Jlens verbalization'], ['logit_lens', 14, 'Logit lens verbalization']]) {
    const titleButton = label(stacks, title, x, (layerCount - 1) * .255 + 1.21, 0, 'stack-title');
    titleButton.onclick = () => onLayer(selected, kind);
    shadow(stacks, x, -(layerCount - 1) * .255 - .61, 0, 15, 12);
    for (let index = 0; index < layerCount; index++) {
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: '#f2f3f5', roughness: .55, transparent: true }));
      mesh.position.set(x, (index - (layerCount - 1) / 2) * .51, 0);
      mesh.rotation.y = -.12;
      mesh.userData = { kind, index, baseX: x, baseY: mesh.position.y, targetColor: color(0) };
      const word = label(mesh, '', kind === 'jlens' ? 6.4 : -6.4, 0, 2.2, 'layer-word');
      word.hidden = true;
      word.tabIndex = -1;
      word.dataset.layer = index;
      word.dataset.method = kind;
      word.onclick = () => onLayer(index, kind);
      mesh.userData.word = word;
      const hoverShadow = shadow(mesh, 0, -.22, 0, 9, 5);
      hoverShadow.material.opacity = 0;
      mesh.userData.shadow = hoverShadow;
      mesh.userData.hover = createBlockHover(mesh, new THREE.Vector3(10.4, .45, 6));
      stacks.add(mesh);
      plates.push(mesh);
      interactive.push(mesh);
    }
  }

  // Glass encloses the full layer. These blocks show component OUTPUT coordinates,
  // not individual attention weights or a reconstructed internal neural circuit.
  const glass = new THREE.Mesh(new RoundedBoxGeometry(11.8, 9, 5.5, 3, .12), new THREE.MeshPhysicalMaterial({ color: '#d8e6f2', transparent: true, opacity: .13, roughness: .12, metalness: .05, depthWrite: false }));
  glass.position.set(-14, 0, 0);
  glass.rotation.y = -.18;
  inside.add(glass);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(11.8, 9, 5.5)), new THREE.LineBasicMaterial({ color: '#b4c5d8', transparent: true, opacity: .65 }));
  edges.position.copy(glass.position);
  edges.rotation.copy(glass.rotation);
  inside.add(edges);
  const layerTitle = label(inside, 'Layer 24', -14, 4.1, 0, 'stage-heading');
  const architectureLabel = label(inside, 'Layer architecture', -14, -4.4, 0, 'dimension-label');
  const residualLabel = label(inside, 'Output after residual addition', -14, -3, 2.5, 'dimension-label');
  line(inside, [[-18, -1.4, 2], [-18, -2.4, 2], [-10, -2.4, 2], [-10, -1.4, 2]]);
  const residualLine = inside.children.at(-1);

  // A canvas texture prints real numbers on a 3D block, not on a side panel.
  function numberBlock(key, title, x, y, width, height, columns, rows, depth = .65, method = 'jlens') {
    title = blockTitles[key];
    width *= 1.4; height *= 1.4; depth *= 1.5;
    const group = new THREE.Group();
    group.position.set(x, y, .8);
    group.rotation.y = -.22;
    group.userData.method = ['attention', 'mlp', 'hidden'].includes(key) ? null : method;
    const material = new THREE.MeshStandardMaterial({ color: '#dce6f2', roughness: .45 });
    const box = new THREE.Mesh(new RoundedBoxGeometry(width, height, depth, 2, .06), material);
    box.userData.stage = { key, method };
    group.add(box);
    interactive.push(box);
    const canvas = document.createElement('canvas');
    canvas.width = columns === 1 ? 192 : 512;
    canvas.height = Math.round(canvas.width * height / width);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const face = new THREE.Mesh(new THREE.PlaneGeometry(width * .97, height * .97), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }));
    face.position.z = depth / 2 + .012;
    group.add(face);
    const titleButton = label(group, title, 0, -height / 2 - .55, depth / 2, 'component-label');
    titleButton.tutorialBounds = () => tutorialBounds([box], camera, renderer.domElement);
    titleButton.onclick = () => onNumber(key, group.userData.method || selectedKind);
    
    const surfaceShadow = shadow(group, 0, -height / 2 - .2, -.1, width * 1.9, 2.5);
    inside.add(group);
    const stage = { key, method, title: titleButton, group, box, face, canvas, texture, columns, rows, shadow: surfaceShadow, baseY: y, order: stages.length };
    stage.hover = createBlockHover(group, new THREE.Vector3(width, height, depth));
    stages.push(stage);
    return stage;
  }
  const attention = numberBlock('attention', 'Attention', -16.5, 0, 3.4, 3.5, 4, 4, 1.8);
  // Small raised cells preserve the volumetric head-grid silhouette without fake values.
  for (let i = 0; i < 4; i++) {
    const rib = new THREE.Mesh(new THREE.BoxGeometry(.7, 3.5, 1.6), new THREE.MeshStandardMaterial({ color: '#d5e1ee', roughness: .5 }));
    rib.position.set(-1.25 + i * .83, 0, -.4);
    attention.group.add(rib);
  }
  const mlp = numberBlock('mlp', 'MLP', -11.7, 0, 3.3, 3.6, 4, 4, 1.4);
  for (const [x, height] of [[-1.9, 2.6], [-2.5, 1.7], [1.9, 2.6], [2.5, 1.7]]) {
    const rib = new THREE.Mesh(new THREE.BoxGeometry(.25, height, .9), new THREE.MeshStandardMaterial({ color: '#c5d6e9', roughness: .5 }));
    rib.position.set(x, 0, -.3);
    mlp.group.add(rib);
  }
  numberBlock('hidden', 'Hidden state', -5.8, 0, 1.55, 5.5, 1, 8);
  const hiddenSize = label(inside, 'Hidden state values', -5.8, -4, 1, 'dimension-label');
  const techniqueTitle = label(inside, 'Jlens verbalization', 7, 5.4, 0, 'stage-heading');
  numberBlock('jacobian', 'Jacobian', -1.8, 0, 3.2, 3.4, 4, 4, 1.2);
  numberBlock('transformed', 'J × h', 1.5, 0, 1.5, 3.4, 1, 6);
  numberBlock('normalized', 'Normalize', 4.3, 0, 1.5, 3.4, 1, 6);
  numberBlock('unembedding', 'Unembedding', 7.8, 0, 3.2, 4.3, 4, 5, 1.2);
  numberBlock('logits', 'Logits', 11.1, 0, 1.55, 3.4, 1, 5);
  numberBlock('softmax', 'Softmax', 14, 0, 2, 3.2, 1, 3);
  numberBlock('normalized', 'Normalize', -1.8, 0, 1.5, 3.4, 1, 6, .65, 'logit_lens');
  numberBlock('unembedding', 'Unembedding', 3, 0, 3.2, 4.3, 4, 5, 1.2, 'logit_lens');
  numberBlock('logits', 'Logits', 7, 0, 1.55, 3.4, 1, 5, .65, 'logit_lens');
  numberBlock('softmax', 'Softmax', 11, 0, 2, 3.2, 1, 3, .65, 'logit_lens');
  const tokenLabels = {}, tokenObjects = {}, scalingLabels = {};
  for (const [method, x] of [['jlens', 18.2], ['logit_lens', 16]]) {
    const element = document.createElement('button');
    element.className = 'word-chips';
    element.setAttribute('aria-label', method === 'jlens' ? 'Inspect Jlens token probabilities' : 'Inspect Logit lens token probabilities');
    element.onclick = () => onNumber('softmax', method);
    const object = new CSS2DObject(element);
    object.position.set(x, 0, 1);
    object.userData.method = method;
    inside.add(object);
    tokenLabels[method] = element;
    tokenObjects[method] = object;
  }
  for (const [method, x] of [['jlens', 9.9], ['logit_lens', 5.4]]) {
    const scaling = label(inside, '÷ 8', x, .65, 1, 'dimension-label');
    scaling.object3D.userData.method = method;
    scalingLabels[method] = scaling.object3D;
  }

  const desktopPositions = new Map(inside.children.map(child => [child, child.position.clone()]));

  function layoutCalculation() {
    const compact = container.clientWidth < 700;
    for (const [object, position] of desktopPositions) object.position.copy(position);
    for (const item of flows) {
      inside.remove(item.group);
      item.wire.dispose();
    }
    flows.length = 0;
    if (compact) {
      glass.position.set(-6, 8, 0);
      edges.position.copy(glass.position);
      residualLine.position.set(8, 8, 0);
      layerTitle.object3D.position.set(-6, 13.1, 0);
      architectureLabel.object3D.position.set(-6, 3.6, 0);
      residualLabel.object3D.position.set(-6, 5, 2.5);
      hiddenSize.object3D.position.set(5, 4, 1);
      techniqueTitle.object3D.position.set(0, 14.5, 0);
      const positions = {
        attention: [-8.5, 8], mlp: [-3.7, 8], hidden: [5, 8],
        jlens: {jacobian: [-6.5, 0], transformed: [0, 0], normalized: [6.5, 0], unembedding: [-6.5, -8], logits: [0, -8], softmax: [6.5, -8]},
        logit_lens: {normalized: [-6.5, 0], unembedding: [0, 0], logits: [6.5, 0], softmax: [0, -8]}
      };
      for (const stage of stages) {
        const position = positions[stage.key] || positions[stage.method][stage.key];
        stage.group.position.set(position[0], position[1], .8);
      }
      for (const method of ['jlens', 'logit_lens']) tokenObjects[method].position.set(0, -14.5, 1);
      scalingLabels.jlens.position.set(-3.2, -7.35, 1);
      scalingLabels.logit_lens.position.set(3.2, .65, 1);
      flow([[-6.7, 8, 1.8], [-6.1, 8, 1.8], [-5.5, 8, 1.8]]);
      flow([[-.8, 8, 1], [1.5, 8, 1], [4.1, 8, 1]]);
      for (const method of ['jlens', 'logit_lens']) {
        flow([[5, 5.1, 1], [5, 3.2, 1], [-6.5, 3.2, 1], [-6.5, 1.9, 1]], method);
      }
      flow([[-4.8, 0, 1], [-2.8, 0, 1], [-.9, 0, 1]], 'jlens');
      flow([[.9, 0, 1], [3.2, 0, 1], [5.6, 0, 1]], 'jlens');
      flow([[6.5, -1.9, 1], [6.5, -4, 1], [-6.5, -4, 1], [-6.5, -5.7, 1]], 'jlens');
      flow([[-4.8, -8, 1], [-2.8, -8, 1], [-.9, -8, 1]], 'jlens');
      flow([[.9, -8, 1], [3.2, -8, 1], [5.4, -8, 1]], 'jlens');
      flow([[6.5, -9.2, 1], [6.5, -11, 1], [0, -11, 1], [0, -11.7, 1]], 'jlens');
      flow([[-5.6, 0, 1], [-3.2, 0, 1], [-1.7, 0, 1]], 'logit_lens');
      flow([[1.7, 0, 1], [3.2, 0, 1], [5.6, 0, 1]], 'logit_lens');
      flow([[6.5, -1.9, 1], [6.5, -4, 1], [0, -4, 1], [0, -6.9, 1]], 'logit_lens');
      flow([[0, -9.2, 1], [0, -10.5, 1], [0, -11.7, 1]], 'logit_lens');
    } else {
      // Two roomy rows use the available height instead of squeezing every block into one line.
      glass.position.set(-10, 5, 0); edges.position.copy(glass.position);
      residualLine.position.set(4, 5, 0);
      layerTitle.object3D.position.set(-10, 10.2, 0);
      architectureLabel.object3D.position.set(-10, -.2, 0);
      residualLabel.object3D.position.set(-10, 1.1, 2.5);
      hiddenSize.object3D.position.set(-1.8, -.45, 1);
      techniqueTitle.object3D.position.set(3, 11.2, 0);
      const positions = {
        attention: [-12.5, 5], mlp: [-7.7, 5], hidden: [-1.8, 5],
        jlens: {jacobian: [4.5, 5], transformed: [10.5, 5], normalized: [-10, -5.5], unembedding: [-3.5, -5.5], logits: [3.5, -5.5], softmax: [9.5, -5.5]},
        logit_lens: {normalized: [4.5, 5], unembedding: [-10, -5.5], logits: [-2.5, -5.5], softmax: [5, -5.5]}
      };
      for (const stage of stages) {
        const point = positions[stage.key] || positions[stage.method][stage.key];
        stage.group.position.set(point[0], point[1], .8);
      }
      tokenObjects.jlens.position.set(9.5, -10.4, 1);
      tokenObjects.logit_lens.position.set(11, -5.5, 1);
      scalingLabels.jlens.position.set(.1, -4.8, 1);
      scalingLabels.logit_lens.position.set(-6.1, -4.8, 1);
      flow([[-10.7, 5, 1.8], [-10.1, 5, 1.8], [-9.5, 5, 1.8]]);
      flow([[-4.8, 5, 1], [-3.9, 5, 1], [-2.7, 5, 1]]);
      flow([[-.9, 5, 1], [1.2, 5, 1], [2.8, 5, 1]], 'jlens');
      flow([[6.2, 5, 1], [7.6, 5, 1], [9.6, 5, 1]], 'jlens');
      flow([[10.5, 3.1, 1], [10.5, -1.4, 1], [-10, -1.4, 1], [-10, -3.6, 1]], 'jlens');
      flow([[-9.1, -5.5, 1], [-7, -5.5, 1], [-5.2, -5.5, 1]], 'jlens');
      flow([[-1.8, -5.5, 1], [.2, -5.5, 1], [2.6, -5.5, 1]], 'jlens');
      flow([[4.4, -5.5, 1], [6.4, -5.5, 1], [8.4, -5.5, 1]], 'jlens');
      flow([[9.5, -6.6, 1], [9.5, -7.6, 1], [9.5, -8.8, 1]], 'jlens');
      flow([[-.9, 5, 1], [1.2, 5, 1], [3.6, 5, 1]], 'logit_lens');
      flow([[4.5, 3.1, 1], [4.5, -1.4, 1], [-10, -1.4, 1], [-10, -3.2, 1]], 'logit_lens');
      flow([[-8.3, -5.5, 1], [-5.8, -5.5, 1], [-3.4, -5.5, 1]], 'logit_lens');
      flow([[-1.6, -5.5, 1], [1.2, -5.5, 1], [3.9, -5.5, 1]], 'logit_lens');
      flow([[6.1, -5.5, 1], [7.5, -5.5, 1], [8.6, -5.5, 1]], 'logit_lens');
    }
    for (const stage of stages) stage.baseY = stage.group.position.y;
    inside.children.forEach(child => { child.visible = !child.userData.method || child.userData.method === selectedKind; });
  }

  function drawNumbers(stage, values) {
    const context = stage.canvas.getContext('2d');
    const width = stage.canvas.width, height = stage.canvas.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#e4edf6';
    context.fillRect(0, 0, width, height);
    context.font = (stage.columns === 1 ? '40px' : '44px') + ' "Viewer Comic", "Comic Sans MS", "Comic Sans", cursive';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    const rows = Math.min(stage.rows, Math.max(1, Math.ceil(values.length / stage.columns)));
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < stage.columns; col++) {
        const x = col * width / stage.columns, y = row * height / rows;
        const value = values[row * stage.columns + col];
        context.strokeStyle = '#b7c9db';
        context.strokeRect(x, y, width / stage.columns, height / rows);
        context.fillStyle = '#243b55';
        context.fillText(Number.isFinite(value) ? (stage.key === 'softmax' ? formatProbability(value) : value.toFixed(2)) : '…', x + width / stage.columns / 2, y + height / rows / 2, width / stage.columns - 12);
      }
    }
    stage.texture.needsUpdate = true;
    // Fade the real value texture in; never interpolate invented numbers.
    stage.face.material.opacity = reduced.matches ? 1 : .72;
  }

  function updateNumbers(frame, weights, strategy) {
    if (!frame || !weights) return;
    currentFrame = { frame, weights, strategy };
    hiddenSize.textContent = weights.hidden_size.toLocaleString() + ' hidden state values';
    for (const method of ['jlens', 'logit_lens']) scalingLabels[method].element.textContent = weights.logits_scaling === 1 ? '' : '÷ ' + weights.logits_scaling;
    const state = frame.layer_values[selected];
    for (const stage of stages) {
      const row = stage.method === 'jlens' ? frame.jlens[strategy][selected] : frame.logit_lens[selected];
      let values = [];
      if (['attention', 'mlp', 'hidden'].includes(stage.key)) values = state[stage.key];
      if (stage.key === 'jacobian') values = weights.jacobians[strategy][String(selected)].slice(0, 4).flatMap(row => row.slice(0, 4));
      if (['transformed', 'normalized'].includes(stage.key)) values = row[stage.key];
      if (stage.key === 'unembedding') values = row.tokens.flatMap(token => weights.unembedding[String(token.id)].slice(0, 4));
      if (stage.key === 'logits') values = row.tokens.map(token => token.logit);
      // These probabilities were normalized across the full vocabulary, not only the shown tokens.
      if (stage.key === 'softmax') values = row.tokens.map(token => token.probability);
      drawNumbers(stage, values);
    }
    for (const method of ['jlens', 'logit_lens']) {
      const row = method === 'jlens' ? frame.jlens[strategy][selected] : frame.logit_lens[selected];
      tokenLabels[method].replaceChildren(...row.tokens.slice(0, 3).map(token => {
        const line = document.createElement('span');
        const word = document.createElement('b');
        word.textContent = JSON.stringify(token.text);
        const probability = document.createElement('small');
        probability.textContent = formatProbability(token.probability);
        line.append(word, probability);
        return line;
      }));
    }
    layerTitle.textContent = 'Layer ' + selected;
    requestRender();
  }

  function setFrame(frame, strategy, weights, first = false) {
    hasData = Boolean(frame);
    currentFrame = frame ? { frame, strategy, weights } : null;
    if (first) { revealElapsed = 0; lastTime = performance.now(); }
    for (const plate of plates) {
      const row = frame && (plate.userData.kind === 'jlens' ? frame.jlens[strategy] : frame.logit_lens)[plate.userData.index];
      plate.userData.targetColor = color(row ? row.difference : 0);
      // Reset the real mesh, including when two prompts produce the same colors.
      if (!frame || first) plate.material.color.copy(color(0));
      const word = plate.userData.word;
      word.hidden = !row || revealElapsed < plate.userData.index * 16;
      word.textContent = row ? `${plate.userData.index} ${JSON.stringify(row.tokens[0].text)}` : '';
      word.title = word.textContent;
    }
    if (detail && frame) updateNumbers(frame, weights, strategy);
    requestRender();
  }
  function setExpanded(value, index = selected, kind = selectedKind) {
    detail = value;
    controls.enabled = value;
    openedAt = performance.now();
    selected = index;
    selectedKind = kind;
    inside.visible = value || inside.visible;
    if (value && currentFrame) updateNumbers(currentFrame.frame, currentFrame.weights, currentFrame.strategy);
    document.body.classList.toggle('view-detail', value);
    transition = { start: performance.now(), from: camera.position.clone(), target: controls.target.clone(), zoom: camera.zoom, to: new THREE.Vector3(value ? 1 : 0, value ? 7 : 14, value ? 48 : 55), look: new THREE.Vector3(value ? 1 : 0, .6, 0), endZoom: value ? 1.06 : 1 };
    // Only the clicked technique is visible, including its words and calculation arrows.
    stacks.visible = !value;
    inside.children.forEach(child => { child.visible = !child.userData.method || child.userData.method === kind; });
    techniqueTitle.textContent = kind === 'jlens' ? 'Jlens verbalization' : 'Logit lens verbalization';
    fitCamera();
    requestRender();
  }
  function fitCamera() {
    const width = container.clientWidth, height = container.clientHeight;
    const usableHeight = Math.max(100, height - 65);
    layoutCalculation();
    const compact = detail && width < 700;
    const halfHeight = detail ? (compact ? Math.max(18 * height / usableHeight, 13 * height / width) : Math.max(12.5 * height / usableHeight, 17.5 * height / width)) : width < 900 ? Math.max(11 * height / usableHeight, 14 * height / width) : Math.max(12 * height / usableHeight, 22 * height / width);
    stackWidth = width < 900 ? .65 : 1.05;
    stackHeight = Math.min(usableHeight - 35, 860) / (20 * height / (2 * halfHeight));
    for (const plate of plates) plate.userData.word.object3D.position.x = (plate.userData.kind === 'jlens' ? 1 : -1) * (width < 900 ? 10.2 : 7.8);
    Object.assign(camera, { top: halfHeight, bottom: -halfHeight, left: -halfHeight * width / height, right: halfHeight * width / height });
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labels.setSize(width, height);
    requestRender();
  }
  function requestRender() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(render);
  }
  function render(now) {
    pending = false;
    if (inspecting) return;
    const seconds = Math.min((now - lastTime) / 1000, .06);
    lastTime = now;
    // Count drawn frames, not time spent loading or waiting for the GPU.
    // A slow first draw must not skip the entire sweep on short answers.
    if (playing && hasData) revealElapsed += Math.min(seconds * 1000, 32);
    const fast = reduced.matches ? 1 : 1 - Math.exp(-seconds * 16);
    const slow = reduced.matches ? 1 : 1 - Math.exp(-seconds * 8);
    let moving = false;
    if (transition) {
      const t = reduced.matches ? 1 : Math.min((now - transition.start) / 850, 1);
      const eased = t * t * (3 - 2 * t);
      camera.position.lerpVectors(transition.from, transition.to, eased);
      controls.target.lerpVectors(transition.target, transition.look, eased);
      camera.zoom = THREE.MathUtils.lerp(transition.zoom, transition.endZoom, eased);
      camera.updateProjectionMatrix();
      if (t === 1) transition = null;
      else moving = true;
    }
    const stackScale = new THREE.Vector3(stackWidth, stackHeight, 1);
    if (stacks.scale.distanceTo(stackScale) > .001) { stacks.scale.lerp(stackScale, slow); moving = true; }
    stacks.position.lerp(new THREE.Vector3(detail ? -12 : 0, detail ? 8 : 0, detail ? -10 : 0), slow);
    for (const plate of plates) {
      const target = plate.userData.targetColor;
      const delayed = hasData && revealElapsed < (plate.userData.index || 0) * 16;
      if (!delayed && Math.abs(plate.material.color.r - target.r) + Math.abs(plate.material.color.g - target.g) + Math.abs(plate.material.color.b - target.b) > .001) { plate.material.color.lerp(target, fast); moving = true; }
      plate.userData.word.hidden = !hasData || delayed;
      if (delayed) moving = true;
      const over = hovered === plate;
      // The moving outline makes playback visible even when logit differences are near zero.
      const reading = playing && !detail && !reduced.matches && Math.abs(plate.userData.index - Math.floor(revealElapsed / 16)) <= 1;
      plate.userData.hover.set(over || reading);
      const highlight = plate.userData.hover.animate(seconds, reduced.matches);
      if (highlight > .005 || reading) moving = true;
      const targetZ = over ? 1.05 : 0;
      if (Math.abs(plate.position.z - targetZ) > .001) { plate.position.z = THREE.MathUtils.lerp(plate.position.z, targetZ, fast); moving = true; }
      plate.userData.shadow.material.opacity = THREE.MathUtils.lerp(plate.userData.shadow.material.opacity, over ? .85 : 0, fast);
    }
    for (const stage of stages) {
      if (stage.face.material.opacity < .999) { stage.face.material.opacity = THREE.MathUtils.lerp(stage.face.material.opacity, 1, fast); moving = true; }
      const waiting = detail && !reduced.matches && now - openedAt < stage.order * 30;
      const targetScale = detail && !waiting ? 1 : .015;
      if (waiting) moving = true;
      if (Math.abs(stage.group.scale.x - targetScale) > .001) { stage.group.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), slow); moving = true; }
      const over = hovered === stage.box;
      stage.hover.set(over); stage.hover.animate(seconds, reduced.matches);
      const float = detail && playing && !reduced.matches ? Math.sin(now / 1100 + stage.order) * .025 : 0;
      const targetY = stage.baseY + float;
      if (Math.abs(stage.group.position.y - targetY) > .001) { stage.group.position.y = THREE.MathUtils.lerp(stage.group.position.y, targetY, fast); moving = true; }
      stage.shadow.material.opacity = THREE.MathUtils.lerp(stage.shadow.material.opacity, over ? .95 : .35, fast);
      stage.box.material.color.lerp(new THREE.Color(over ? '#bbcce0' : '#dce6f2'), fast);
    }
    inside.scale.lerp(new THREE.Vector3(detail ? 1 : .02, detail ? 1 : .02, detail ? 1 : .02), slow);
    if (!detail && inside.scale.x < .03) inside.visible = false;
    for (const item of flows) item.wire.animate(now, reduced.matches);
    controls.update();
    renderer.render(scene, camera);
    labels.render(scene, camera);
    if (moving || (playing && hasData) || (detail && !reduced.matches)) requestRender();
  }
  controls.addEventListener('change', requestRender);
  controls.addEventListener('start', () => { transition = null; });
  reduced.addEventListener('change', () => { controls.enableDamping = !reduced.matches; playing = false; requestRender(); });
  renderer.domElement.addEventListener('keydown', event => {
    if (['+', '=', '-'].includes(event.key)) {
      event.preventDefault();
      camera.zoom = THREE.MathUtils.clamp(camera.zoom * (event.key === '-' ? 1 / 1.1 : 1.1), controls.minZoom, controls.maxZoom);
      camera.updateProjectionMatrix();
      requestRender();
    }
  });
  function pick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(interactive.filter(mesh => mesh.userData.stage ? detail && mesh.parent.visible : !detail), false)[0]?.object || null;
  }
  renderer.domElement.addEventListener('pointermove', event => {
    hovered = pick(event);
    renderer.domElement.style.cursor = hovered ? 'pointer' : 'grab';
    requestRender();
  });
  renderer.domElement.addEventListener('pointerleave', () => { hovered = null; requestRender(); });
  let start;
  renderer.domElement.addEventListener('pointerdown', event => { start = [event.clientX, event.clientY]; });
  renderer.domElement.addEventListener('click', event => {
    if (!hasData || !start || Math.hypot(event.clientX - start[0], event.clientY - start[1]) > 5) return;
    const hit = pick(event);
    if (hit?.userData.stage) {
      const stage = stages.find(stage => stage.box === hit);
      const face = raycaster.intersectObject(stage.face, false)[0];
      if (face) {
        const row = Math.min(stage.rows - 1, Math.floor((1 - face.uv.y) * stage.rows));
        const column = Math.min(stage.columns - 1, Math.floor(face.uv.x * stage.columns));
        const matrix = ['jacobian', 'unembedding'].includes(stage.key);
        onNumber(stage.key, stage.group.userData.method || selectedKind, matrix ? row : row * stage.columns + column, matrix ? column : 0);
      } else onNumber(stage.key, stage.group.userData.method || selectedKind);
    }
    else if (hit) onLayer(hit.userData.index, hit.userData.kind);
  });
  camera.position.set(0, 14, 55);
  controls.target.set(0, .6, 0);
  controls.update();
  new ResizeObserver(fitCamera).observe(container);
  fitCamera();
  function setInspecting(value) {
    inspecting = value;
    if (value) {hovered = null; stages.forEach(stage => {stage.hover.set(false); stage.hover.animate(1, true);});}
    if (!value && currentFrame) updateNumbers(currentFrame.frame, currentFrame.weights, currentFrame.strategy);
    requestRender();
  }
  container.tutorialBounds = () => tutorialBounds(stages.filter(stage => stage.group.visible).map(stage => stage.box), camera, renderer.domElement);
  return { setFrame, setExpanded, updateNumbers, setInspecting, isRevealing: () => hasData && revealElapsed < layerCount * 16 + 160, setPlaying: value => { playing = value; if (!value) revealElapsed = Infinity; requestRender(); }, camera, plates, controls, renderer, inside, stacks, stages, color };
}
