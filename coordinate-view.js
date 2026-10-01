import { tutorialBounds } from './tutorial-bounds.js?v=tour-fit-31';
import * as THREE from 'three';
import { theme } from './theme.js';
import { createBlockHover } from './block-hover.js?v=selected-value-17';
import { createFlowWire } from './flow-wire.js?v=value-flow-13';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { coordinateData } from './coordinate-data.js?v=transformer-terms-34';
import { blockTitles, describeBlock, vectorValueNames, formatProbability, tokenProbabilityLabel, blockTrainingDescriptions } from './block-labels.js?v=transformer-terms-34';

// The original layer meshes move into this camera view and return unchanged on close.
export function createCoordinateView(host, onClose) {
  host.innerHTML = `<div class="coordinate-heading"><strong>Inside Antares</strong><span id="coordinate-title"></span><button class="tutorial-launch" data-tutorial> Tutorial </button><button id="coordinate-close">Back to layers</button></div>
    <div class="coordinate-canvas" tabindex="0" role="region" aria-label="3D component. Drag to inspect vector and matrix values. Arrow keys select a vector or matrix value."></div>
    <article class="coordinate-explanation" aria-live="polite"><h2></h2><p id="coordinate-description"></p><details id="coordinate-training" hidden><summary>How this matrix is fitted</summary><p></p></details><p id="coordinate-selection"></p><p id="coordinate-equation"></p><p id="coordinate-meaning"></p><p id="coordinate-total"></p></article>`;
  const surface = host.querySelector('.coordinate-canvas');
  const info = name => host.querySelector('#coordinate-' + name);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-16, 16, 10, -10, .1, 150);
  camera.position.set(0, 5, 32); camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  surface.append(renderer.domElement);
  const callout = document.createElement('output');
  callout.className = 'value-callout'; callout.hidden = true; surface.append(callout);
  let selectedBlockKey;
  const labels = new CSS2DRenderer();
  labels.domElement.className = 'coordinate-labels';
  surface.append(labels.domElement);
  scene.add(new THREE.HemisphereLight('#ffffff', '#bdc9d9', 2.4));
  const light = new THREE.DirectionalLight('#ffffff', 2.2); light.position.set(-10, 20, 25); scene.add(light);
  const wires = [];
  let connectionsDirty = true;
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let blocks = [], context, data, selectedRow = 0, selectedColumn = 0, rowStart = 0, columnStart = 0;
  let lastFrame = performance.now();
  let opened = false, pending = false, started = 0, previousFocus, drag, dragged = false;
  const count = 4;

  function clearButtons(block) {
    for (const object of block.buttons) { object.element.remove(); object.removeFromParent(); }
    block.buttons = [];
  }
  function drawBlock(block) {
    clearButtons(block);
    const display = block.display;
    if (!display) return;
    const {stage, size} = block;
    const canvas = stage.canvas, ctx = canvas.getContext('2d');
    const rows = display.values.length, columns = display.values[0].length;
    ctx.fillStyle = block.foreground ? '#cee2f6' : '#e0e2e5'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = `${columns > 1 ? 32 : 28}px "Viewer Comic", cursive`;
    for (let i = 0; i < rows; i++) for (let j = 0; j < columns; j++) {
      const row = display.startRow + i, column = display.startColumn + j;
      const chosen = display.kind === 'input' ? row === selectedColumn : display.kind === 'output' ? row === selectedRow : row === selectedRow && column === selectedColumn;
      const x = j * canvas.width / columns, y = i * canvas.height / rows;
      if (chosen && block.foreground) { ctx.fillStyle = theme['accent-soft']; ctx.fillRect(x + 3, y + 3, canvas.width / columns - 6, canvas.height / rows - 6); }
      ctx.fillStyle = block.foreground ? theme.title : '#737b86';
      const value = display.values[i][j];
      if (display.names) {
        ctx.font = '20px "Viewer Comic", cursive';
        ctx.fillText(JSON.stringify(display.names[i]), canvas.width * .28, y + canvas.height / rows / 2, canvas.width * .5);
        ctx.fillText((block.key === 'softmax' ? formatProbability(value) : Number(value).toPrecision(3)), canvas.width * .78, y + canvas.height / rows / 2, canvas.width * .4);
      } else ctx.fillText(Number(value).toPrecision(3), x + canvas.width / columns / 2, y + canvas.height / rows / 2, canvas.width / columns - 28);
      if (!block.foreground) continue;
      // Invisible HTML targets provide keyboard access; the visible numbers are on the solid's texture.
      const button = document.createElement('button');
      button.className = 'coordinate-tile'; button.dataset.kind = display.kind; button.dataset.row = row; button.dataset.column = column;
      button.classList.toggle('selected', chosen); button.setAttribute('aria-pressed', String(chosen));
      button.setAttribute('aria-label', block.key === 'softmax' ? tokenProbabilityLabel(display.names[i], value) : `${data.input && display.kind === 'weight' ? 'Matrix value' : vectorValueNames[block.key]} [${row}, ${column}]: ${value}`); button.title = String(value);
      button.onpointerenter = button.onfocus = () => hoverValue(block, row, column);
      button.onpointerleave = button.onblur = () => {block.hover.set(false); block.valueHover.set(false); render();};
      button.onclick = () => { if (!dragged) choose(display.kind === 'input' ? selectedRow : row, display.kind === 'input' ? row : display.kind === 'output' ? selectedColumn : column, false, block.key); };
      const object = new CSS2DObject(button);
      object.position.set((j + .5) / columns * size.x - size.x / 2, size.y / 2 - (i + .5) / rows * size.y, size.z / 2 + .04);
      object.userData.cellWidth = size.x / columns; object.userData.cellHeight = size.y / rows;
      stage.group.add(object); block.buttons.push(object);
    }
    stage.texture.needsUpdate = true;
  }
  // Selection is separate from hover, so it stays visible after the pointer leaves.
  function markSelection(block) {
    const d = block.display, rows = d.values.length, columns = d.values[0].length;
    const row = d.kind === 'input' ? selectedColumn : selectedRow;
    const column = d.kind === 'input' || d.kind === 'output' ? 0 : selectedColumn;
    const visible = block.foreground && row >= d.startRow && row < d.startRow + rows && column >= d.startColumn && column < d.startColumn + columns;
    const width = block.size.x / columns, height = block.size.y / rows;
    block.valueSelection.set(visible,
      new THREE.Vector3((column - d.startColumn + .5) * width - block.size.x / 2, block.size.y / 2 - (row - d.startRow + .5) * height, 0),
      new THREE.Vector3(width, height, block.size.z + .08));
    block.selection = {row, column, visible};
  }
  function placeCallout() {
    const selected = blocks.find(block => block.key === selectedBlockKey);
    if (!selected?.selection.visible) {callout.hidden = true; return;}
    const {row, column} = selected.selection, d = selected.display;
    const value = d.values[row - d.startRow][column - d.startColumn];
    const index = d.kind === 'weight' ? 'Matrix value [' + row + ', ' + column + ']' : vectorValueNames[selected.key] + ' ' + row;
    callout.textContent = selected.key === 'softmax'
      ? tokenProbabilityLabel(d.names[row - d.startRow], value)
      : blockTitles[selected.key] + '\n' + index + ' = ' + Number(value).toPrecision(6);
    callout.dataset.block = selected.key; callout.dataset.row = row; callout.dataset.column = column;
    callout.hidden = false;
    const width = callout.offsetWidth, height = callout.offsetHeight;
    const bounds = blocks.map(block => {
      const points = [];
      for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
        const v = block.group.localToWorld(new THREE.Vector3(x * block.size.x / 2, y * block.size.y / 2, z * block.size.z / 2)).project(camera);
        points.push({x: (v.x + 1) * surface.clientWidth / 2, y: (1 - v.y) * surface.clientHeight / 2});
      }
      return {left: Math.min(...points.map(p => p.x)), right: Math.max(...points.map(p => p.x)), top: Math.min(...points.map(p => p.y)), bottom: Math.max(...points.map(p => p.y))};
    });
    const target = bounds[blocks.indexOf(selected)];
    const overlaps = (a, b) => a.left < b.right + 10 && a.right > b.left - 10 && a.top < b.bottom + 10 && a.bottom > b.top - 10;
    const surfaceRect = surface.getBoundingClientRect();
    // Include labels and the explanation, not just the mesh faces, as obstacles.
    for (const element of host.querySelectorAll('.coordinate-stage, .coordinate-explanation')) {
      const rect = element.getBoundingClientRect();
      bounds.push({left: rect.left - surfaceRect.left, right: rect.right - surfaceRect.left, top: rect.top - surfaceRect.top, bottom: rect.bottom - surfaceRect.top});
    }
    const middleX = (target.left + target.right - width) / 2, middleY = (target.top + target.bottom - height) / 2;
    const candidates = [[target.right + 16, middleY], [target.left - width - 16, middleY], [middleX, target.top - height - 18], [middleX, target.bottom + 18]];
    // A nearby free slot is preferable to covering a neighboring matrix or vector.
    for (let y = 8; y + height < surface.clientHeight; y += 20) {
      for (const x of [middleX, target.right + 16, target.left - width - 16, 8, surface.clientWidth - width - 8]) candidates.push([x, y]);
    }
    const valid = candidates.map(([x, y]) => ({left: x, top: y, right: x + width, bottom: y + height}))
      .filter(rect => rect.left >= 8 && rect.top >= 8 && rect.right <= surface.clientWidth - 8 && rect.bottom <= surface.clientHeight - 8 && !bounds.some(obstacle => overlaps(rect, obstacle)));
    valid.sort((a, b) => Math.hypot(a.left - middleX, a.top - middleY) - Math.hypot(b.left - middleX, b.top - middleY));
    const place = valid[0];
    // Tiny viewports may temporarily have no free slot during the zoom transition.
    callout.hidden = !place;
    if (place) {callout.style.left = place.left + 'px'; callout.style.top = place.top + 'px';}
  }

  function hoverValue(block, row, column) {
    blocks.forEach(item => {item.hover.set(item === block); item.valueHover.set(false);});
    const d = block.display, width = block.size.x / d.values[0].length, height = block.size.y / d.values.length;
    // Outline the value's volume through the full depth of the original solid.
    block.valueHover.set(true, new THREE.Vector3((column - d.startColumn + .5) * width - block.size.x / 2, block.size.y / 2 - (row - d.startRow + .5) * height, 0), new THREE.Vector3(width, height, block.size.z + .06));
    render();
  }
  function updateValues() {
    connectionsDirty = true;
    const inputKey = context.key === 'jacobian' ? 'hidden' : 'normalized';
    const outputKey = context.key === 'jacobian' ? 'transformed' : 'logits';
    for (const block of blocks) {
      block.active = block.key === context.key;
      block.role = block.active ? 'main' : data.input && block.key === inputKey ? 'input' : data.input && block.key === outputKey ? 'output' : 'background';
      block.foreground = block.role !== 'background';
      const other = coordinateData(context.frame, context.weights, context.strategy, context.layer, context.method, block.key);
      let values, startRow = 0, startColumn = 0, kind = 'value';
      if (block.active) {
        startRow = rowStart; startColumn = columnStart; kind = data.input ? 'weight' : 'value';
        values = data.values.slice(rowStart, rowStart + count).map(row => row.slice(columnStart, columnStart + count));
      } else if (block.role === 'input') {
        startRow = columnStart; kind = 'input'; values = data.input.slice(columnStart, columnStart + count).map(value => [value]);
      } else if (block.role === 'output') {
        startRow = rowStart; kind = 'output'; values = data.output.slice(rowStart, rowStart + count).map(value => [value]);
      } else values = other.values.slice(0, count).map(row => row.slice(0, count));
      block.display = {values, startRow, startColumn, kind, names: !other.input && other.names ? other.names.slice(startRow, startRow + values.length) : null};
      drawBlock(block);
      markSelection(block);
    }
    const explanation = data.explain(selectedRow, selectedColumn);
    host.querySelector('.coordinate-explanation h2').textContent = blockTitles[context.key];
    info('description').textContent = describeBlock(context.key, context.model, context.layer);
    info('training').hidden = !blockTrainingDescriptions[context.key];
    info('training').querySelector('p').textContent = blockTrainingDescriptions[context.key] || '';
    info('selection').textContent = data.input ? `Matrix value [${selectedRow}, ${selectedColumn}]` : '';
    for (const name of ['equation', 'meaning', 'total']) info(name).textContent = explanation[name] || '';
    render();
  }
  function targets() {
    const background = blocks.filter(block => !block.foreground);
    const narrow = surface.clientWidth < 650;
    for (const block of blocks) {
      block.from = block.stage.group.position.clone(); block.fromScale = block.stage.group.scale.clone();
      const index = background.indexOf(block);
      const center = narrow ? 0 : 3;
      block.target = block.role === 'main' ? new THREE.Vector3(center, narrow ? 1.6 : -1.2, 2)
        : block.role === 'input' ? new THREE.Vector3(narrow ? -4.4 : center - 8.5, narrow ? -8.2 : -1.2, 1)
        : block.role === 'output' ? new THREE.Vector3(narrow ? 4.4 : center + 8.5, narrow ? -8.2 : -1.2, 1)
        : new THREE.Vector3(narrow ? (index % 4 - 1.5) * 3.5 : (index - (background.length - 1) / 2) * 3.1 + 2.5, narrow ? 10.6 - Math.floor(index / 4) * 2.5 : 6.4, -8);
      const scale = block.active ? Math.min(12.6 / block.size.y, 12.6 / block.size.x) : block.foreground ? (narrow ? 4.8 : 8.4) / block.size.y : .46;
      block.targetScale = new THREE.Vector3(scale, scale, scale);
      block.stage.title.setAttribute('aria-current', String(block.active));
      for (const child of block.stage.group.children.filter(child => child.isMesh)) {
        child.material.color?.set(block.foreground ? child === block.stage.face ? '#ffffff' : '#bfd8f1' : '#bcc0c6');
        if (child === block.stage.shadow) child.material.opacity = block.foreground ? .3 : .12;
      }
    }
    connectionsDirty = true; started = performance.now() - (reduced.matches ? 700 : 0); render();
  }
  function setKey(key, row = 0, column = 0) {
    context.key = key; selectedBlockKey = key; host.dataset.stage = key;
    data = coordinateData(context.frame, context.weights, context.strategy, context.layer, context.method, key);
    selectedRow = Math.min(row, data.values.length - 1); selectedColumn = Math.min(column, data.values[0].length - 1);
    rowStart = Math.max(0, Math.min(selectedRow - 1, data.values.length - count));
    columnStart = Math.max(0, Math.min(selectedColumn - 1, data.values[0].length - count));
    updateValues(); targets();
  }
  function choose(row, column, focus = false, blockKey = context.key) {
    selectedBlockKey = blockKey;
    selectedRow = Math.max(0, Math.min(data.values.length - 1, row)); selectedColumn = Math.max(0, Math.min(data.values[0].length - 1, column));
    if (selectedRow < rowStart || selectedRow >= rowStart + count) rowStart = Math.max(0, Math.min(selectedRow, data.values.length - count));
    if (selectedColumn < columnStart || selectedColumn >= columnStart + count) columnStart = Math.max(0, Math.min(selectedColumn, data.values[0].length - count));
    updateValues();
    if (focus) { labels.render(scene, camera); host.querySelector('.coordinate-tile.selected[data-kind="weight"], .coordinate-tile.selected[data-kind="value"]')?.focus({preventScroll:true}); }
  }
  function connections() {
    const paths = [];
    const port = (block, side) => block.group.localToWorld(new THREE.Vector3(side * block.size.x / 2, 0, block.size.z / 2));
    // Preserve the complete sequence while its original blocks move into focus.
    for (let i = 0; i < blocks.length - 1; i++) {
      const source = blocks[i], target = blocks[i + 1];
      const a = port(source, 1), b = port(target, -1);
      const c = a.clone().add(new THREE.Vector3(1.3, 0, 0));
      const d = b.clone().add(new THREE.Vector3(-1.3, 0, 0));
      // Long return wires pass behind the values instead of through their faces.
      if (b.x < a.x || Math.abs(b.y - a.y) > 2) { c.z -= 3; d.z -= 3; }
      paths.push({curve: new THREE.CubicBezierCurve3(a, c, d, b), color: '#7194ba', radius: .075});
    }
    if (data?.input) {
      const main = blocks.find(block => block.active);
      const endpoint = (block, row, column, side) => {
        const d = block.display, size = block.size;
        return block.group.localToWorld(new THREE.Vector3((column - d.startColumn + .5 + side * .32) / d.values[0].length * size.x - size.x / 2, size.y / 2 - (row - d.startRow + .8) / d.values.length * size.y, size.z / 2 + .1));
      };
      for (const role of ['input', 'output']) {
        const block = blocks.find(item => item.role === role);
        if (!block) continue;
        const middle = endpoint(main, selectedRow, selectedColumn, role === 'input' ? -1 : 1);
        const edge = endpoint(block, role === 'input' ? selectedColumn : selectedRow, 0, role === 'input' ? 1 : -1);
        const bend = edge.clone().lerp(middle, .5); bend.z = Math.max(edge.z, middle.z) + 3;
        // Contributions flow from input to weight, then from weight to output.
        const points = role === 'input' ? [edge, bend, middle] : [middle, bend, edge];
        paths.push({curve: new THREE.CatmullRomCurve3(points), color: theme.accent, radius: .035});
      }
    }
    while (wires.length > paths.length) wires.pop().dispose();
    paths.forEach((path, index) => {
      if (!wires[index] || wires[index].group.userData.color !== path.color) {
        wires[index]?.dispose();
        wires[index] = createFlowWire(path.curve, path.color, path.radius);
        wires[index].group.userData.color = path.color; scene.add(wires[index].group);
      } else wires[index].update(path.curve);
    });
    connectionsDirty = false;
  }
  function sizeValueTargets() {
    for (const block of blocks) for (const button of block.buttons) {
      const center = button.position;
      const left = block.group.localToWorld(center.clone().add(new THREE.Vector3(-button.userData.cellWidth / 2, 0, 0))).project(camera);
      const right = block.group.localToWorld(center.clone().add(new THREE.Vector3(button.userData.cellWidth / 2, 0, 0))).project(camera);
      const top = block.group.localToWorld(center.clone().add(new THREE.Vector3(0, button.userData.cellHeight / 2, 0))).project(camera);
      const bottom = block.group.localToWorld(center.clone().add(new THREE.Vector3(0, -button.userData.cellHeight / 2, 0))).project(camera);
      button.element.style.width = Math.max(16, Math.abs(right.x-left.x) * surface.clientWidth * .43) + 'px';
      button.element.style.height = Math.max(16, Math.abs(top.y-bottom.y) * surface.clientHeight * .40) + 'px';
    }
  }
  function resize() {
    if (!opened) return;
    const width = Math.max(1, surface.clientWidth), height = Math.max(1, surface.clientHeight);
    const halfHeight = width < 650 ? Math.max(13, 7.4 * height / width) : Math.max(9.8, 17 * height / width);
    Object.assign(camera, {left: -halfHeight * width / height, right: halfHeight * width / height, top: halfHeight, bottom: -halfHeight});
    camera.updateProjectionMatrix(); renderer.setSize(width, height); labels.setSize(width, height);
    if (blocks.length) targets();
    render();
  }
  function render() { if (!opened || pending) return; pending = true; requestAnimationFrame(animate); }
  function animate(now) {
    pending = false; if (!opened) return;
    const seconds = Math.min((now - lastFrame) / 1000, .06); lastFrame = now;
    const t = reduced.matches ? 1 : Math.min(1, (now - started) / 700), eased = t * t * (3 - 2 * t);
    for (const block of blocks) {
      block.stage.group.position.lerpVectors(block.from, block.target, eased);
      block.stage.group.scale.lerpVectors(block.fromScale, block.targetScale, eased);
      block.stage.group.rotation.y = -.25;
      const glow = block.hover.animate(seconds, reduced.matches);
      block.valueHover.animate(seconds, reduced.matches);
      block.valueSelection.animate(seconds, reduced.matches);
      block.stage.shadow.material.opacity = (block.foreground ? .3 : .12) + glow * .65;
    }
    scene.updateMatrixWorld(true);
    if (t < 1 || connectionsDirty) { connections(); connectionsDirty = t < 1; }
    wires.forEach(wire => wire.animate(now, reduced.matches));
    sizeValueTargets(); renderer.render(scene, camera); labels.render(scene, camera); placeCallout();
    if (t < 1 || !reduced.matches) render();
  }
  function open(options) {
    host.querySelector('.coordinate-heading strong').textContent = 'Inside ' + (options.modelName || 'Antares');
    info('close').textContent = `Back to Layer ${options.layer} structure`;
    context = options; previousFocus = document.activeElement; opened = true; host.hidden = false;
    document.body.classList.add('coordinate-open'); document.querySelector('header').inert = true; document.querySelector('#main').inert = true;
    info('title').textContent = `${options.method === 'jlens' ? 'Jlens' : 'Logit lens'} · Layer ${options.layer}${options.method === 'jlens' ? ' · Strategy ' + options.strategy.slice(-1) : ''}`;
    blocks = options.stages.filter(stage => !stage.group.userData.method || stage.group.userData.method === options.method).map(stage => {
      stage.box.geometry.computeBoundingBox(); const size = stage.box.geometry.boundingBox.getSize(new THREE.Vector3());
      const restore = {parent:stage.group.parent, position:stage.group.position.clone(), scale:stage.group.scale.clone(), rotation:stage.group.rotation.clone(), tutorialBounds:stage.title.tutorialBounds, labels:[], materials:[]};
      stage.group.traverse(child => {
        if (child.isMesh) { restore.materials.push([child, child.material]); child.material = child.material.clone(); }
        if (child.element) { restore.labels.push([child.element, child.element.hidden, child.element.className, child.element.onclick, child.element.parentNode]); child.element.hidden = child.element !== stage.title; }
      });
      scene.add(stage.group);
      stage.title.className = 'coordinate-stage'; stage.title.dataset.stage = stage.key; stage.title.textContent = blockTitles[stage.key]; stage.title.onclick = () => setKey(stage.key);
      stage.title.tutorialBounds = () => tutorialBounds([stage.box], camera, renderer.domElement);
      const hover = createBlockHover(stage.group, size), valueHover = createBlockHover(stage.group, size);
      const valueSelection = createBlockHover(stage.group, size, .20);
      stage.title.onpointerenter = stage.title.onfocus = () => {hover.set(true); render();};
      stage.title.onpointerleave = stage.title.onblur = () => {hover.set(false); render();};
      return {key:stage.key, stage, group:stage.group, size, restore, buttons:[], hover, valueHover, valueSelection};
    });
    setKey(options.key, options.row || 0, options.column || 0); resize(); info('close').focus({preventScroll:true});
  }
  function close() {
    if (!opened) return;
    opened = false;
    for (const block of blocks) {
      clearButtons(block); block.hover.dispose(); block.valueHover.dispose(); block.valueSelection.dispose(); const {stage, restore} = block;
      stage.title.tutorialBounds = restore.tutorialBounds;
      stage.title.onpointerenter = stage.title.onpointerleave = stage.title.onfocus = stage.title.onblur = null;
      for (const [mesh, material] of restore.materials) { mesh.material.dispose(); mesh.material = material; }
      for (const [element, hidden, className, onclick, parent] of restore.labels) { element.hidden = hidden; element.className = className; element.onclick = onclick; delete element.dataset.stage; element.removeAttribute('aria-current'); parent?.append(element); }
      restore.parent.add(stage.group); stage.group.position.copy(restore.position); stage.group.scale.copy(restore.scale); stage.group.rotation.copy(restore.rotation);
    }
    wires.splice(0).forEach(wire => wire.dispose());
    blocks = []; callout.hidden = true; host.hidden = true; document.body.classList.remove('coordinate-open');
    document.querySelector('header').inert = false; document.querySelector('#main').inert = false;
    onClose?.(); previousFocus?.focus({preventScroll:true});
  }
  info('close').onclick = close;
  host.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.stopPropagation(); close(); }
    if (event.key === 'Tab') {
      const buttons = [...host.querySelectorAll('button')].filter(button => !button.hidden && button.tabIndex >= 0);
      if (event.shiftKey && document.activeElement === buttons[0]) { event.preventDefault(); buttons.at(-1).focus(); }
      else if (!event.shiftKey && document.activeElement === buttons.at(-1)) { event.preventDefault(); buttons[0].focus(); }
    }
  });
  surface.addEventListener('keydown', event => {
    const steps = {ArrowLeft:[0,-1], ArrowRight:[0,1], ArrowUp:[-1,0], ArrowDown:[1,0]};
    if (steps[event.key]) { event.preventDefault(); choose(selectedRow + steps[event.key][0], selectedColumn + steps[event.key][1], true); }
  });
  surface.addEventListener('click', event => {
    if (dragged || event.target.closest('button')) return;
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, 1 - (event.clientY - bounds.top) / bounds.height * 2);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(blocks.flatMap(block => [block.stage.face, block.stage.box]), false)[0];
    if (!hit) return;
    const block = blocks.find(block => hit.object === block.stage.face || hit.object === block.stage.box);
    if (!block.foreground) { setKey(block.key); return; }
    if (hit.object !== block.stage.face) return;
    const d = block.display, row = d.startRow + Math.min(d.values.length - 1, Math.floor((1-hit.uv.y) * d.values.length)), column = d.startColumn + Math.min(d.values[0].length - 1, Math.floor(hit.uv.x * d.values[0].length));
    choose(block.role === 'input' ? selectedRow : row, block.role === 'input' ? row : block.role === 'output' ? selectedColumn : column, false, block.key);
  });
  surface.addEventListener('pointerdown', event => { if (event.button === 0) { drag = {x:event.clientX,y:event.clientY,row:rowStart,column:columnStart}; dragged = false; } });
  surface.addEventListener('pointermove', event => {
    if (!drag) {
      if (event.target.closest('button')) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, 1 - (event.clientY - bounds.top) / bounds.height * 2);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(blocks.flatMap(block => [block.stage.face, block.stage.box]), false)[0];
      const block = hit && blocks.find(item => item.stage.face === hit.object || item.stage.box === hit.object);
      blocks.forEach(item => {item.hover.set(item === block); item.valueHover.set(false);});
      if (block?.foreground && hit.object === block.stage.face) {
        const d = block.display;
        hoverValue(block, d.startRow + Math.min(d.values.length - 1, Math.floor((1-hit.uv.y) * d.values.length)), d.startColumn + Math.min(d.values[0].length - 1, Math.floor(hit.uv.x * d.values[0].length)));
      }
      render(); return;
    }
    const dx = event.clientX-drag.x, dy = event.clientY-drag.y; if (Math.hypot(dx,dy)<8) return;
    dragged = true; surface.setPointerCapture(event.pointerId);
    const row = Math.max(0,Math.min(data.values.length-count,drag.row-Math.trunc(dy/30))), column = Math.max(0,Math.min(data.values[0].length-count,drag.column-Math.trunc(dx/30)));
    if(row!==rowStart||column!==columnStart) { rowStart=row;columnStart=column;choose(row,column); }
  });
  surface.addEventListener('pointerleave', () => {blocks.forEach(block => {block.hover.set(false); block.valueHover.set(false);}); render();});
  surface.addEventListener('pointerup', () => {drag=null;setTimeout(()=>{dragged=false;},0);});
  surface.addEventListener('pointercancel', () => {drag=null;});
  surface.addEventListener('wheel', event => {event.preventDefault();choose(selectedRow+Math.sign(event.deltaY),selectedColumn+Math.sign(event.deltaX));},{passive:false});
  new ResizeObserver(resize).observe(surface); reduced.addEventListener('change', () => { if (reduced.matches) started = performance.now() - 700; render(); });
  return {open,close,choose,camera,renderer,get blocks(){return blocks;}, get wires(){return wires;}};
}
