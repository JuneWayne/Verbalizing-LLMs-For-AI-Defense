import { createLayerPlot } from './layer-plot.js?v=prompt-levels-35';
import { unpackRecording } from './recording-codec.js?v=1';
import { formatProbability } from './block-labels.js?v=transformer-terms-34';
import { modelSummary } from './model-copy.js?v=transformer-terms-34';
import { createFrameLoader } from './frame-loader.js?v=seek-replay-24';
import { createCoordinateView } from './coordinate-view.js?v=transformer-terms-34';
import { createScene } from './scene.js?v=prompt-levels-35';

const ui = Object.fromEntries([...document.querySelectorAll('[id]')].map(element => [element.id, element]));
const modelResponse = await fetch('./models.json', {cache: 'no-store'});
if (!modelResponse.ok) throw new Error('The model list could not be loaded.');
const models = await modelResponse.json();
const modelId = new URLSearchParams(location.search).get('model') || 'antares-1b';
const activeModel = models.find(model => model.id === modelId);
if (!activeModel) throw new Error('Choose a model from the landing page.');
// Each available model loads only its own catalog and recorded layer states.
const hasRecordings = Boolean(activeModel.viewer);
const modelLink = document.querySelector('.model-return');
modelLink.textContent = '← Inside ' + (modelId === 'antares-1b' ? 'Antares' : activeModel.name);
modelLink.href = './index.html?model=' + encodeURIComponent(modelId);
document.title = 'Inside ' + activeModel.name + ' | Injection detection';
document.querySelector('.subtitle').textContent = hasRecordings ? 'Recorded examples' : activeModel.layers + ' layers';
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let levelIndex = null, traceHistory = null;
let catalog, recording, weights, weightsPromise, frames = [], checkFrame, result, classificationIndex = -1;
let strategy = 'strategy1', selected = 24, selectedKind = 'jlens', expanded = false, timer = null, requestNumber = 0;
let download, coordinates, resumeReplay = false, frameLoader, displayedFrame, frameRequest = 0, playbackRun = 0;
const recordingCache = new Map();
const layerPlots = {jlens: createLayerPlot(ui['jlens-plot'], 'jlens', selectPlotLayer, seekPlotToken), logit_lens: createLayerPlot(ui['logit-plot'], 'logit_lens', selectPlotLayer, seekPlotToken)};
let plotPanel = 'stacks';
function selectPlotLayer(index) {
  selected = index; ui.layer.value = index;
  showFrame(Number(ui.step.value)); showHistory(); updateLevelTable();
}
function seekPlotToken(index) { stopReplay(); showFrame(Math.min(index, Number(ui.step.max))); }
function showPlotPanel(panel, smooth = true) {
  plotPanel = panel;
  const viewport = ui['comparison-scroll'];
  const left = panel === 'jlens' ? 0 : panel === 'stacks' ? ui.scene.offsetLeft : viewport.scrollWidth - viewport.clientWidth;
  viewport.scrollTo({left, behavior: smooth && !reducedMotion.matches ? 'smooth' : 'instant'});
}
ui['show-jlens-plot'].onclick = () => showPlotPanel('jlens');
ui['show-stacks'].onclick = () => showPlotPanel('stacks');
ui['show-logit-plot'].onclick = () => showPlotPanel('logit_lens');
// Remember the panel reached by touchpad or touch, not only navigation buttons.
ui['comparison-scroll'].addEventListener('scrollend', () => {
  if (expanded) return;
  const viewport = ui['comparison-scroll'];
  const positions = [['jlens', 0], ['stacks', ui.scene.offsetLeft], ['logit_lens', viewport.scrollWidth - viewport.clientWidth]];
  plotPanel = positions.sort((a, b) => Math.abs(viewport.scrollLeft - a[1]) - Math.abs(viewport.scrollLeft - b[1]))[0][0];
});
// Horizontal touchpad movement navigates panels; ordinary wheel zoom stays in 3D.
ui['comparison-scroll'].addEventListener('wheel', event => {
  if (expanded || !(Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey)) return;
  event.preventDefault(); event.stopPropagation();
  ui['comparison-scroll'].scrollLeft += event.deltaX || event.deltaY;
}, {capture: true, passive: false});
// Load the shared family before drawing text into the 3D number textures.
await document.fonts.load('14px "Viewer Comic"');
const view = createScene(ui.scene, selectLayer, inspectNumber, activeModel.recordedLayers?.length || (hasRecordings ? 39 : activeModel.layers));

// Keep the projected labels out of the fixed controls, including at browser zoom.
function layoutScene() {
  const scale = Number(getComputedStyle(document.body).zoom) || 1;
  const header = document.querySelector('header');
  const actions = document.querySelector('.scene-actions');
  const narrow = innerWidth / scale < 704;
  document.body.classList.toggle('comparison-compact', innerWidth / scale < 1000);
  const legend = document.querySelector('.legend');
  document.documentElement.style.setProperty('--compact-playback-bottom', (legend.offsetHeight + parseFloat(getComputedStyle(legend).bottom) + 16) + 'px');
  // Move the action column below a wrapped model title.
  const brand = document.querySelector('.brand');
  actions.style.top = Math.max(narrow ? 88 : 90, brand.offsetTop + brand.offsetHeight + 16) + 'px';
  const top = (narrow ? Math.max(header.offsetHeight, actions.offsetTop + actions.offsetHeight) : header.offsetHeight) + 16;
  const bottom = Math.max(...['.playback', '.legend'].map(selector => {
    const element = document.querySelector(selector);
    return element.offsetHeight + parseFloat(getComputedStyle(element).bottom);
  })) + 16;
  // Short screens scroll vertically instead of hiding the graph behind controls.
  const minimumHeight = top + bottom + 48 + 300;
  const scrollPage = innerHeight / scale < minimumHeight;
  document.body.classList.toggle('comparison-short', scrollPage);
  document.body.style.minHeight = scrollPage ? minimumHeight + 'px' : '';
  document.documentElement.style.setProperty('--scene-top', top + 'px');
  document.documentElement.style.setProperty('--scene-bottom', bottom + 'px');
  document.documentElement.style.setProperty('--scene-left', narrow ? '0px' : (actions.offsetLeft + actions.offsetWidth + 16) + 'px');
}
const comparisonObserver = new ResizeObserver(() => {
  ui['comparison-scroll'].style.setProperty('--comparison-width', ui['comparison-scroll'].clientWidth + 'px');
  showPlotPanel(plotPanel, false);
});
comparisonObserver.observe(ui['comparison-scroll']);
const layoutObserver = new ResizeObserver(layoutScene);
for (const selector of ['header', '.scene-actions', '.playback', '.legend']) layoutObserver.observe(document.querySelector(selector));
window.addEventListener('resize', layoutScene);
layoutScene();

function fitDialog(dialog) {
  const scale = Number(getComputedStyle(document.body).zoom) || 1;
  dialog.style.maxHeight = `${innerHeight / scale - 32}px`;
  dialog.style.maxWidth = `${innerWidth / scale - 32}px`;
}
window.addEventListener('resize', () => document.querySelectorAll('dialog[open]').forEach(fitDialog));

async function openDialog(id) {
  stopReplay();
  if (id === 'levels-dialog') updateLevelTable();
  if (id === 'tokens-dialog' && await showClassification() === false) return;
  fitDialog(ui[id]);
  ui[id].showModal();
}
function closeDialog(dialog) {
  if (!dialog.open || dialog.classList.contains('closing')) return;
  // Exit motion finishes before closing, so native dialog focus restoration still works.
  if (reducedMotion.matches) { dialog.close(); return; }
  dialog.classList.add('closing');
  setTimeout(() => { dialog.classList.remove('closing'); dialog.close(); }, 150);
}
document.querySelectorAll('[data-dialog]').forEach(button => { button.onclick = () => openDialog(button.dataset.dialog); });
document.querySelectorAll('[data-close]').forEach(button => { button.onclick = () => closeDialog(button.closest('dialog')); });
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeDialog(dialog); });
  // Close only when the click starts and ends outside the card, not after dragging its contents.
  let pressedOutside = false;
  const outside = event => {
    const bounds = dialog.getBoundingClientRect();
    return event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
  };
  dialog.addEventListener('pointerdown', event => { pressedOutside = event.target === dialog && outside(event); });
  dialog.addEventListener('click', event => {
    if (pressedOutside && event.target === dialog && outside(event)) closeDialog(dialog);
    pressedOutside = false;
  });
});

function currentFrame() {
  return displayedFrame;
}
async function showClassification() {
  if (!recording) return;
  const index = classificationIndex;
  if (index < 0) return;
  stopReplay();
  const request = requestNumber;
  try {
    await frameLoader.ensure(index);
  } catch (error) {
    if (request === requestNumber && error.name !== 'AbortError') ui.status.textContent = 'Could not load the classification step. Select the layer again to retry.';
    return false;
  }
  if (request !== requestNumber) return false;
  ui.position.value = 'response';
  showFrame(index);
}
async function selectLayer(index, kind = selectedKind, classification = true) {
  if (!recording) return;
  const request = requestNumber;
  if (!weights && !await loadInspectionWeights()) return;
  if (request !== requestNumber) return;
  selected = index;
  selectedKind = kind;
  stopReplay();
  if (classification && await showClassification() === false) return;
  if (request !== requestNumber) return;
  expanded = true;
  ui['back-to-layers'].textContent = 'Back to layers';
  ui.layer.value = selected;
  view.setExpanded(true, selected, kind);
  ui['comparison-scroll'].scrollLeft = 0;

  showFrame(Number(ui.step.value));
  showHistory();
}
function showOverview() {
  coordinates?.close();
  if (expanded) view.setExpanded(false);
  expanded = false;
  ui['back-to-layers'].textContent = 'Back to models';
  showPlotPanel('stacks', false);
}
async function inspectNumber(key, method = 'jlens', row = key === 'jacobian' ? 5 : 0, column = key === 'jacobian' ? 12 : 0) {
  if (!recording) return;
  const request = requestNumber;
  if (!weights && !await loadInspectionWeights()) return;
  if (request !== requestNumber) return;
  stopReplay();
  if (!coordinates) coordinates = createCoordinateView(ui['coordinate-view'], () => view.setInspecting(false));
  view.setInspecting(true);
  coordinates.open({frame: currentFrame(), weights, strategy, layer: selected, method, stages: view.stages,
    key, row, column, modelName: activeModel.name, model: activeModel});
}

function showFrame(index, first = false) {
  if (!recording) return;
  const separate = ui.position.value === 'check';
  const frame = separate ? checkFrame : frames[index];
  if (!frame) return false;
  const displayRequest = ++frameRequest;
  if (!separate) { ui.step.value = index; frameLoader.focus(index); }
  if (!frame.layer_values) {
    ui['step-label'].textContent = `Loading token ${index + 1} of ${frames.length}`;
    ui.step.setAttribute('aria-busy', 'true');
    const request = requestNumber;
    frameLoader.ensure(index).then(() => {
      if (request === requestNumber && displayRequest === frameRequest) showFrame(index, first);
    }).catch(error => {
      if (request !== requestNumber || displayRequest !== frameRequest || error.name === 'AbortError') return;
      stopReplay();
      ui.status.textContent = 'Could not load this response step. Move the slider to retry.';
    });
    return false;
  }
  displayedFrame = frame;
  ui.step.setAttribute('aria-busy', 'false');
  if (!separate) ui.step.value = index;
  ui.step.disabled = separate;
  ui.play.disabled = separate;
  const labelPosition = !separate && index === classificationIndex;
  ui['classification-status'].hidden = classificationIndex >= 0;
  ui['step-label'].textContent = separate ? 'Separate label check' : `Token ${frame.step} of ${frames.length}${labelPosition ? ' · final label' : ''}`;
  // Show the token actually generated; a tied top score can list another word first.
  const generatedToken = frame.generated_token?.text ?? frame.model.tokens.find(token => token.id === frame.token_id)?.text ?? frame.text.slice(index ? frames[index - 1].text.length : 0);
  ui['token-preview'].textContent = separate ? '' : JSON.stringify(generatedToken);
  ui['token-preview'].title = ui['token-preview'].textContent;
  ui['position-note'].textContent = separate ? 'Extra forward pass with an added classification cue. This is not the original answer.' : labelPosition ? (result.format === 'label is not aligned with single-token scoring' ? 'State before the generated label. Its token form differs from the two scored label tokens.' : `State immediately before the generated ${result.decision} label.`) : classificationIndex < 0 ? 'No classification label generated. These words predict the response, not a label.' : 'Response token, not the final label.';
  ui['layer-title'].textContent = `Layer ${selected} · ${selectedKind === 'jlens' ? 'Jlens' : 'Logit lens'}`;
  ui['j-score'].closest('section').hidden = selectedKind !== 'jlens';
  ui['l-score'].closest('section').hidden = selectedKind !== 'logit_lens';
  for (const [prefix, row] of [['j', frame.jlens[strategy][selected]], ['l', frame.logit_lens[selected]], ['m', frame.model]]) {
    ui[`${prefix}-score`].textContent = `${row.difference > 0 ? '+' : ''}${row.difference.toFixed(3)}`;
    ui[`${prefix}-labels`].textContent = `injection: ${row.label_logits.injection.toFixed(3)}\nsafe: ${row.label_logits.safe.toFixed(3)}`;
    ui[`${prefix}-tokens`].replaceChildren(...row.tokens.map(token => {
      const item = document.createElement('li');
      item.textContent = `${JSON.stringify(token.text)}  ${token.logit.toFixed(3)}  (${formatProbability(token.probability)})`;
      return item;
    }));
  }
  view.setFrame(frame, strategy, weights, first);
  for (const [method, plot] of Object.entries(layerPlots)) plot.update(frame, activeModel.recordedLayers, strategy, selected, traceHistory || frameLoader.frames, catalog.system_level ? 'Level ' + catalog.system_level.level : '');
  if (!separate) frameLoader.prefetch(index);
  return true;
}
function showHistory() {
  if (!recording) return;
  ui['layer-history'].textContent = frameLoader.summaries.map(frame => {
    if (frame.layer_words) {
      const words = selectedKind === 'jlens' ? frame.layer_words.jlens[strategy] : frame.layer_words.logit_lens;
      return `${frame.step}. ${JSON.stringify(words[selected])}`;
    }
    const rows = selectedKind === 'jlens' ? frame.jlens[strategy] : frame.logit_lens;
    return `${frame.step}. ${JSON.stringify(rows[selected].tokens[0].text)}`;
  }).join('\n');
}
function stopReplay() {
  resumeReplay = false;
  clearTimeout(timer);
  timer = null;
  playbackRun++;
  frameRequest++;
  ui.step.setAttribute('aria-busy', 'false');
  ui.play.firstElementChild.textContent = '▶';
  ui.play.setAttribute('aria-label', 'Replay response');
  ui['play-state'].textContent = 'Replay';
  view.setPlaying(false);
}
function startReplay(fromStart = false) {
  if (!recording) return;
  stopReplay();
  // A VS Code preview can become hidden while a prompt loads.
  if (document.hidden) { resumeReplay = true; return; }
  ui.position.value = 'response';
  // Keep playback at the selected token, ending at the actual generated label.
  const lastIndex = classificationIndex >= 0 ? classificationIndex : frames.length - 1;
  let index = fromStart || Number(ui.step.value) >= lastIndex ? 0 : Number(ui.step.value);
  const run = playbackRun;
  const loader = frameLoader;
  ui.step.value = index;
  loader.focus(index);
  ui.play.firstElementChild.textContent = 'Ⅱ';
  ui.play.setAttribute('aria-label', 'Pause replay');
  ui['play-state'].textContent = 'Pause';
  // Reset a cached first frame before enabling animation, avoiding a flash of the full stack.
  const ready = Boolean(frames[index]?.layer_values);
  if (ready) { showFrame(index, fromStart || index === 0); index++; }
  view.setPlaying(true);

  // Await one frame at a time. Seeking, pausing, or changing prompts invalidates this run.
  async function advance() {
    if (run !== playbackRun) return;
    if (index > lastIndex) {
      if (!view.isRevealing()) { stopReplay(); return; }
    } else {
      try {
        loader.focus(index);
        if (!frames[index]?.layer_values) {
          ui['play-state'].textContent = 'Loading…';
          ui['step-label'].textContent = `Loading token ${index + 1} of ${frames.length}`;
          ui.step.setAttribute('aria-busy', 'true');
          await loader.ensure(index);
        }
        if (run !== playbackRun) return;
        ui['play-state'].textContent = 'Pause';
        showFrame(index, index === 0);
        index++;
      } catch (error) {
        if (run !== playbackRun) return;
        stopReplay();
        ui.status.textContent = 'Could not load this response step. Press play to retry.';
        return;
      }
    }
    timer = setTimeout(advance, 72);
  }
  timer = setTimeout(advance, ready ? 72 : 0);
}
function sourceName(source) {
  return /cami|office email pairs/i.test(source) ? 'synthetic office scenario emails' : source;
}
function promptWording(text) {
  return text.replace(/cami|office email pairs/gi, 'office scenario prompts').replace(/\bemails?\b/gi, 'prompts');
}
function showStrategy() {
  if (!catalog) return;
  const training = catalog.strategies.find(row => row.id === strategy);
  ui['strategy-title'].textContent = `${training.name}: ${promptWording(training.prompt_format)}`;
  ui['strategy-description'].textContent = training.id === 'strategy1'
      ? 'Jlens was fitted on 100 short prompts, balanced between injection and safe, without an added classification instruction. Fitting the lens does not change the language model’s weights.'
      : 'Short prompt excerpts are wrapped in a classification instruction, with a label cue at the end. Jlens is fitted to the model’s internal states for these prompts. The language model’s weights are unchanged.';
  ui['strategy-caveat'].textContent = '';
  ui['strategy-caveat'].hidden = true;
  ui['training-source'].textContent = training.id === 'strategy2' ? 'Training data: synthetic office scenario emails.' : `Training data: ${sourceName(training.dataset)}.`;
  showFrame(Number(ui.step.value));
  showHistory();
}

async function loadInspectionWeights() {
  try {
    if (!weightsPromise) weightsPromise = compressed(catalog.weights.file, catalog.weights.sha256).catch(error => { weightsPromise = null; throw error; });
    const pending = weightsPromise;
    const loaded = await pending;
    if (pending !== weightsPromise) return null;
    weights = loaded;
    return weights;
  } catch (error) {
    ui.status.textContent = 'Could not load layer details. Select the layer again to retry.';
    return null;
  }
}

async function compressed(file, hash, signal, retry = true) {
  // Hash-versioned files can be cached safely; verify their contents on every read.
  const url = new URL(file, location.href);
  url.searchParams.set('v', hash);
  const response = await fetch(url, { signal, cache: retry ? 'force-cache' : 'reload' });
  if (!response.ok) throw new Error(`Download returned ${response.status}.`);
  if (!('DecompressionStream' in window)) throw new Error('Use a current browser to open these recordings.');
  const data = await response.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', data);
  const actual = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
  if (actual !== hash) {
    if (retry) return compressed(file, hash, signal, false);
    const error = new Error('The recording was updated. Please try again shortly.');
    error.code = 'RECORDING_CHANGED';
    throw error;
  }
  return unpackRecording(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('gzip'))).json());
}
async function readCatalog(signal) {
  const selectedLevel = levelIndex?.levels.find(level => level.id === ui['system-level'].value);
  const response = await fetch(selectedLevel?.catalog || activeModel.catalog || './catalog.json', { signal, cache: 'no-store' });
  if (!response.ok) throw new Error('The example list could not be loaded.');
  const latest = await response.json();
  if (latest.schema_version !== 2 || !Array.isArray(latest.examples)) throw new Error('The example list is incomplete.');
  return latest;
}

async function loadExample(retry = true) {
  stopReplay();
  const previousPanel = expanded ? 'stacks' : plotPanel;
  showOverview();
  showPlotPanel(previousPanel, false);
  const request = ++requestNumber;
  download?.abort();
  download = new AbortController();
  recording = null;
  traceHistory = null;
  frames = [];
  displayedFrame = null;
  frameLoader = null;
  frameRequest++;
  classificationIndex = -1;
  ui['classification-status'].hidden = true;
  checkFrame = null;
  view.setFrame(null, strategy, weights);
  for (const plot of Object.values(layerPlots)) plot.update(null, [], strategy, selected);
  const example = catalog.examples.find(row => row.id === ui.example.value);
  ui['scene-message'].hidden = false;
  ui['scene-message'].textContent = example ? 'Loading recorded vectors and matrices…' : 'Choose a prompt to start';
  for (const name of ['play', 'step', 'position', 'layer', 'expand', 'inspect', 'view-email', 'full-response', 'system-prompt']) ui[name].disabled = true;
  for (const name of ['response', 'decision', 'layer-history', 'prompt', 'position-note']) ui[name].textContent = '';
  for (const prefix of ['j', 'l', 'm']) for (const suffix of ['score', 'labels', 'tokens']) ui[`${prefix}-${suffix}`].replaceChildren();
  if (!example) { ui.status.textContent = ''; ui['step-label'].textContent = 'Choose a prompt'; ui['token-preview'].textContent = ''; return; }
  ui.status.textContent = 'Loading recording';
  ui.email.textContent = example.text;
  ui.expected.textContent = `Dataset label: ${example.label}`;
  ui.source.textContent = `Source: ${sourceName(example.source)}`;
  try {
    // Playback needs the recording, not the separate inspection matrices.
    const saved = recordingCache.get(example.sha256) || await compressed(example.file, example.sha256, download.signal);
    if (request !== requestNumber) return;
    if (saved.schema_version !== 2 || saved.settings.example.id !== example.id || saved.events.at(-1)?.type !== 'done') throw new Error('Incomplete or mismatched recording.');
    recordingCache.delete(example.sha256);
    recordingCache.set(example.sha256, saved);
    if (recordingCache.size > 2) recordingCache.delete(recordingCache.keys().next().value);
    recording = saved;
    const start = saved.events.find(event => event.type === 'start');
    frameLoader = createFrameLoader(saved, compressed, download.signal);
    // Compact score histories let plots seek without downloading every earlier matrix.
    if (example.trace) compressed(example.trace.file, example.trace.sha256, download.signal).then(trace => {
      if (request !== requestNumber) return;
      if (trace.frames.length !== frameLoader.frames.length || trace.layers.join(',') !== activeModel.recordedLayers.join(',')) throw new Error('Trace does not match this recording.');
      traceHistory = trace.frames;
      if (displayedFrame) for (const [method, plot] of Object.entries(layerPlots)) plot.update(displayedFrame, activeModel.recordedLayers, strategy, selected, traceHistory, catalog.system_level ? 'Level ' + catalog.system_level.level : '');
    }).catch(error => { if (request === requestNumber && error.name !== 'AbortError') ui.status.textContent = 'The full score history could not load. Playback still shows available token scores.'; });
    frames = frameLoader.frames;
    await frameLoader.ensure(0);
    if (request !== requestNumber) return;
    checkFrame = saved.events.find(event => event.type === 'classification_check');
    result = saved.events.at(-1);
    classificationIndex = frames.findIndex(frame => frame.step === result.classification_step);
    // A valid generated label may use a bare token instead of the spaced scoring token.
    // Locate its actual recorded boundary without changing the response or label scores.
    if (classificationIndex < 0 && ['injection', 'safe'].includes(result.decision)) {
      const labelStart = frames.at(-1).text.lastIndexOf(result.decision);
      if (labelStart >= 0) classificationIndex = frames.findIndex(frame => frame.text.length > labelStart);
    }
    ui.layer.replaceChildren(...start.layers.map((layer, index) => new Option(layer, index)));
    selected = Math.min(selected, start.layers.length - 1);
    ui.layer.value = selected;
    ui.response.textContent = frames.at(-1).text;
    ui.decision.textContent = result.decision || 'No classification label generated';
    ui['generation-note'].textContent = start.generation ? 'This recording uses sampling. These are raw model probabilities, before sampling adjustments; the generated token need not have the highest score.' : 'This recording uses greedy decoding: each generated token has the highest model score.';
    // Read the system message from this recording, not from today's Python settings.
    const systemMessage = start.system_prompt.match(/<\|start_of_role\|>system<\|end_of_role\|>([\s\S]*?)<\|end_of_text\|>/);
    ui.prompt.textContent = start.system_message || (systemMessage ? systemMessage[1].trim() : 'The system message is unavailable for this recording.');
    ui.position.querySelector('[value="check"]').disabled = !checkFrame;
    ui.position.value = 'response';
    // Do not let the slider select an end token after the final classification.
    ui.step.max = classificationIndex >= 0 ? classificationIndex : frames.length - 1;
    ui.step.value = 0;
    for (const name of ['play', 'step', 'position', 'layer', 'expand', 'inspect', 'view-email', 'full-response', 'system-prompt']) ui[name].disabled = false;
    showFrame(0, true);
    showHistory();
    ui['scene-message'].hidden = true;
    ui.status.textContent = `${catalog.examples.length} saved examples · Recorded vectors and matrices`;
    // Reduced motion removes decorative movement, not the response playback itself.
    startReplay(true);
  } catch (error) {
    if (request !== requestNumber || error.name === 'AbortError') return;
    // The page may stay open while recordings change. Refresh the catalog once,
    // then verify the newly downloaded files again. Never bypass the hash check.
    if (retry && error.code === 'RECORDING_CHANGED') {
      ui['scene-message'].textContent = 'Loading the updated recording…';
      try {
        const latest = await readCatalog(download.signal);
        if (request !== requestNumber) return;
        catalog = latest;
        weightsPromise = null;
        weights = null;
        return await loadExample(false);
      } catch (refreshError) {
        if (request !== requestNumber || refreshError.name === 'AbortError') return;
        error = refreshError;
      }
    }
    ui.status.textContent = 'Could not load this recording';
    ui['scene-message'].textContent = `${error.message} Choose another prompt or refresh to retry.`;
  }
}

function updateLevelTable() {
  const rows = levelIndex?.comparisons.filter(row => row.example_id === ui.example.value) || [];
  ui['compare-levels'].disabled = rows.length === 0;
  const example = catalog?.examples.find(example => example.id === ui.example.value);
  ui['levels-layer'].textContent = example ? `${promptWording(example.title)} · Expected label: ${example.label} · Layer ${activeModel.recordedLayers[selected]}` : '';
  ui['levels-results'].replaceChildren();
  const number = value => Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : 'Unavailable';
  for (const row of rows) {
    const level = levelIndex.levels.find(level => level.id === row.level_id);
    const tr = document.createElement('tr');
    const label = row.generated_label || 'No valid label';
    for (const value of ['Level ' + level.level, label + (row.stop === 'token limit' ? ' (token limit)' : ''), number(row.model_difference), number(row.jlens?.strategy1[selected]), number(row.jlens?.strategy2[selected]), number(row.logit_lens?.[selected])]) {
      const cell = document.createElement('td'); cell.textContent = value; tr.append(cell);
    }
    ui['levels-results'].append(tr);
  }
}
function populatePrompts(previous = '') {
  ui.example.replaceChildren(new Option('Choose a prompt', ''));
  for (const group of [...new Set(catalog.examples.map(example => example.group))]) {
    const options = document.createElement('optgroup'); options.label = promptWording(group);
    for (const example of catalog.examples.filter(example => example.group === group)) options.append(new Option(promptWording(example.title), example.id));
    ui.example.append(options);
  }
  ui.example.value = catalog.examples.some(example => example.id === previous) ? previous : '';
  ui.example.disabled = false;
}
async function changeSystemLevel() {
  const previousPanel = expanded ? 'stacks' : plotPanel;
  stopReplay(); showOverview(); showPlotPanel(previousPanel, false);
  const previous = ui.example.value;
  const request = ++requestNumber;
  download?.abort(); download = new AbortController();
  recording = null; traceHistory = null; displayedFrame = null;
  weights = null; weightsPromise = null;
  view.setFrame(null, strategy, null);
  for (const plot of Object.values(layerPlots)) plot.update(null, [], strategy, selected);
  ui['system-level'].disabled = true;
  for (const name of ['example', 'play', 'step', 'inspect', 'view-email', 'full-response', 'system-prompt', 'compare-levels']) ui[name].disabled = true;
  try {
    const next = await readCatalog(download.signal);
    if (request !== requestNumber) return;
    catalog = next;
    populatePrompts(previous);
    await loadExample();
    updateLevelTable();
  } catch (error) {
    if (request === requestNumber && error.name !== 'AbortError') {
      ui['scene-message'].hidden = false;
      ui['scene-message'].textContent = 'This system prompt recording could not load. Select another level to retry.';
    }
  } finally { ui['system-level'].disabled = false; }
}
ui['system-level'].onchange = changeSystemLevel;
ui.example.onchange = () => { loadExample(); updateLevelTable(); };
document.querySelectorAll('[name="strategy"]').forEach(input => { input.onchange = () => { strategy = input.value; showStrategy(); }; });
ui.layer.onchange = () => { selected = Number(ui.layer.value); showFrame(Number(ui.step.value)); showHistory(); if (expanded) selectLayer(selected, selectedKind, false); };
ui.expand.onclick = () => { ui['tokens-dialog'].close(); selectLayer(selected, selectedKind, false); };
ui['back-to-layers'].onclick = () => {
  if (expanded) showOverview();
  else location.href = modelLink.href;
};
ui.position.onchange = () => { stopReplay(); showFrame(Number(ui.step.value)); };
ui.step.oninput = () => { stopReplay(); showFrame(Number(ui.step.value)); };
ui.play.onclick = () => timer ? stopReplay() : startReplay();
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    const wasPlaying = timer !== null || resumeReplay;
    stopReplay();
    resumeReplay = wasPlaying;
  } else if (resumeReplay) startReplay();
});
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !document.querySelector('dialog[open]')) showOverview(); });
reducedMotion.addEventListener('change', stopReplay);

if (hasRecordings) {
  if (activeModel.system_levels) {
    const response = await fetch(activeModel.system_levels, {cache: 'no-store'});
    if (!response.ok) throw new Error('System prompt comparisons could not load.');
    levelIndex = await response.json();
    if (levelIndex.model_id !== activeModel.id || levelIndex.levels.length !== 5) throw new Error('System prompt comparisons do not match this model.');
    ui['system-level'].replaceChildren(...levelIndex.levels.map(level => new Option('Level ' + level.level + ': ' + level.title, level.id)));
  }
  catalog = await readCatalog();
  ui['system-level'].disabled = !levelIndex;
  const summary = modelSummary(activeModel);
  ui.architecture.replaceChildren(Object.assign(document.createElement('summary'), {textContent: 'Model architecture'}), ...[summary.architecture, summary.attention, summary.normalization, summary.scaling, summary.vocabulary, summary.layers].map(text => Object.assign(document.createElement('p'), {textContent: text})));
  populatePrompts();
  ui['color-bar'].style.background = `linear-gradient(to right, ${Array.from({ length: 81 }, (_, index) => `${view.color(-20 + index / 2).getStyle()} ${index * 100 / 80}%`).join(', ')})`;
  showStrategy();
  ui['scene-message'].textContent = 'Choose a prompt to start';
  ui['step-label'].textContent = 'Choose a prompt';
} else {
  // Empty stacks belong to this model. No Antares catalog or values are loaded.
  ui.example.replaceChildren(new Option('No saved prompts', ''));
  ui['scene-message'].hidden = true;
  ui['step-label'].textContent = '';
  document.querySelector('[data-dialog="training-dialog"]').disabled = true;
  document.querySelectorAll('[name="strategy"]').forEach(input => {input.disabled = true;});
  ui.status.textContent = activeModel.name + ': no saved prompt recordings.';
  ui['color-bar'].style.background = 'linear-gradient(to right, #709dd7, #f2f3f5, #eb9587)';
}
