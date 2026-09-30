// One guide serves both pages and adapts when a layer or block is opened.
const launch = document.createElement('button');
launch.className = 'tutorial-launch';
launch.dataset.tutorial = '';
launch.textContent = 'Tutorial';
document.body.append(launch);

const overlay = document.createElement('section');
overlay.className = 'tutorial-overlay';
overlay.hidden = true;
overlay.setAttribute('aria-label', 'Website tutorial');
overlay.innerHTML = `<div class="tutorial-ring" aria-hidden="true"></div>
  <div class="tutorial-bubble" role="region" aria-label="Tutorial step">
    <div class="tutorial-top"><span data-count></span><button data-close-tour aria-label="Close tutorial">×</button></div>
    <div class="tutorial-copy" aria-live="polite" aria-atomic="true"><h2></h2><p></p></div>
    <div class="tutorial-controls"><button data-back>Back</button><button data-target>Try it</button><button data-next>Next</button></div>
  </div>`;
document.body.append(overlay);
const bubble = overlay.querySelector('.tutorial-bubble');
const ring = overlay.querySelector('.tutorial-ring');
const back = overlay.querySelector('[data-back]');
const next = overlay.querySelector('[data-next]');
const tryIt = overlay.querySelector('[data-target]');
let steps = [], index = 0, trigger, context, active = false, frame = 0, lastTarget;

function visible(element) {
  return element && element.getClientRects().length && getComputedStyle(element).visibility !== 'hidden' && !element.closest('[inert]');
}
function target(selector) {
  return [...document.querySelectorAll(selector)].find(visible);
}
function currentContext() {
  if (visible(document.querySelector('#coordinate-view'))) return 'block';
  if (document.body.classList.contains('view-detail')) return 'layer';
  return document.querySelector('#example') ? 'viewer' : 'landing';
}
function guideSteps(view) {
  if (view === 'landing') return [
    ['.model-dock nav', 'Choose a model', 'Use the left and right arrows or swipe across the models. The selected model moves to the front.'],
    ['#choose-model', 'Find a model by name', 'Click Choose model to open the list. Select a name to bring its architecture to the front.'],
    ['.legend', 'Read the architecture', 'The colors identify attention, MLP, and normalization blocks. Each model has its own layer count and architecture.'],
    ['#architecture', 'Open the model', 'Click the model at the front to open its recorded prompts and layer predictions. You can also focus the model and press Enter.']
  ];
  if (view === 'block') return [
    ['.coordinate-stage[aria-current="true"]', 'Follow this block', 'The selected block moves forward. The connected blocks show where its input comes from and where its output goes.'],
    ['.coordinate-tile.selected[data-kind="weight"], .coordinate-tile.selected[data-kind="value"]', 'Select a number', 'Click a number on the 3D block. Its box stays highlighted, and the label beside it shows the selected vector or matrix value. Use arrow keys to move between numbers.'],
    ['.coordinate-explanation', 'Read the calculation', 'The description explains this block. Select a different number to update the equation and see how that number is used.'],
    ['#coordinate-training', 'How Jlens is fitted', 'Open How this matrix is fitted to read how training prompts are used to calculate and average the Jacobian matrices.'],
    ['.coordinate-stage:not([aria-current="true"])', 'Inspect another block', 'Click another block or its title to follow the next part of the calculation. Drag across a matrix to see more saved numbers.'],
    ['#coordinate-close', 'Return to the layer', 'Click here to return to this layer’s structure. Then use Back to layers to return to the two stacks.']
  ];
  if (view === 'layer') return [
    ['.component-label', 'Look inside this layer', 'Click a block or its title to bring the original 3D block closer and inspect its recorded numbers.'],
    ['#scene', 'Follow the calculation', 'Follow the connecting lines from the hidden state to the token probabilities. Jlens first estimates a final hidden state using its Jacobian matrix. Logit lens reads the current hidden state directly.'],
    ['#back-to-layers', 'Return to the stacks', 'Click Back to layers to compare other layers or open the other lens.']
  ];
  const loaded = !document.querySelector('#inspect').disabled;
  return [
    ['#example', 'Choose a prompt', 'Select a prompt to replay a saved model response. The colored layers update as the model generates each token.'],
    ['.strategy-switch', 'Compare Jlens strategies', 'Switch between the two fitted Jlens matrices. Both strategies read the same model response and hidden states.'],
    ['[data-dialog="training-dialog"]', 'Read about training', 'Click About training to see the selected strategy’s dataset and prompt format.'],
    ...(loaded ? [
      ['#view-email', 'Read the input', 'Click View prompt to see the exact prompt, its dataset label, and its source.'],
      ['#system-prompt', 'Read the instructions', 'Click System prompt to see the instructions the model received for this recording.'],
      ['.playback-row', 'Control the replay', 'Pause the response or drag the slider to a token. Press play to continue from that position.'],
      ['#main > .legend', 'Read the colors', 'Blue favors safe and orange favors injection. The number compares the injection token score with the safe token score at the selected response position.'],
      ['#inspect', 'Inspect a layer', 'Click a layer in either stack, or use Inspect layers and Look inside. The detail view shows only the lens you selected. Start Tutorial there for a guide to its blocks.'],
      ['#full-response', 'Check the actual response', 'Click Full response to read the generated answer and its final label. Layer predictions and the model’s generated label are shown separately.']
    ] : [
      ['#example', 'Start exploring', 'Choose a prompt first. Once it loads, click Tutorial again for the playback controls, layer predictions, and recorded numbers.']
    ])
  ];
}

function position() {
  if (!active) return;
  const anchor = target(steps[index][0]);
  if (!anchor) { ring.hidden = true; return; }
  const rect = anchor.tutorialBounds?.() || anchor.getBoundingClientRect();
  const padding = 6;
  // Clip only at the viewport edges. Never shrink a wide section into a small circle.
  const left = Math.max(3, rect.left - padding), top = Math.max(3, rect.top - padding);
  const right = Math.min(innerWidth - 3, rect.right + padding), bottom = Math.min(innerHeight - 3, rect.bottom + padding);
  ring.hidden = right <= left || bottom <= top;
  if (ring.hidden) return;
  const width = right - left, height = bottom - top;
  const style = getComputedStyle(anchor);
  const radius = anchor.tutorialBounds || anchor.matches('.coordinate-tile') ? 10
    : Math.min(height / 2, width / 2, (parseFloat(style.borderTopLeftRadius) || 10) + padding);
  Object.assign(ring.style, {left: left + 'px', top: top + 'px', width: width + 'px', height: height + 'px', borderRadius: radius + 'px'});

  const gap = 16, edge = 12;
  bubble.style.maxHeight = (innerHeight - edge * 2) + 'px';
  let cardWidth = bubble.offsetWidth, cardHeight = bubble.offsetHeight;
  // Preserve the full card height; move it instead of squeezing its controls into a gap.
  const clampX = x => Math.max(edge, Math.min(x, innerWidth - cardWidth - edge));
  const clampY = y => Math.max(edge, Math.min(y, innerHeight - cardHeight - edge));
  const middleX = (left + right - cardWidth) / 2, middleY = (top + bottom - cardHeight) / 2;
  const candidates = [
    [right + gap, middleY], [left - cardWidth - gap, middleY],
    [middleX, bottom + gap], [middleX, top - cardHeight - gap]
  ].map(([x, y]) => ({x: clampX(x), y: clampY(y)}));
  for (const place of candidates) {
    const overlap = Math.max(0, Math.min(place.x + cardWidth, right) - Math.max(place.x, left))
      * Math.max(0, Math.min(place.y + cardHeight, bottom) - Math.max(place.y, top));
    place.score = overlap * 10000 + Math.hypot(place.x - middleX, place.y - middleY);
  }
  candidates.sort((a, b) => a.score - b.score);
  Object.assign(bubble.style, {left: candidates[0].x + 'px', top: candidates[0].y + 'px'});
}
function showStep() {
  const [selector, title, description] = steps[index];
  overlay.querySelector('h2').textContent = title;
  overlay.querySelector('p').textContent = description;
  overlay.querySelector('[data-count]').textContent = `${index + 1} of ${steps.length}`;
  back.disabled = index === 0;
  next.textContent = index === steps.length - 1 ? 'Finish' : 'Next';
  const anchor = target(selector);
  tryIt.hidden = !anchor?.matches('button:not(:disabled), select:not(:disabled), [tabindex], details');
  lastTarget = selector;
  overlay.dataset.highlightTarget = selector;
  // Center off-screen sections inside the scrollable inspector, including projected blocks.
  if (anchor) {
    const rect = anchor.tutorialBounds?.() || anchor.getBoundingClientRect();
    if (rect.top < 12 || rect.bottom > innerHeight - 12) {
      if (context === 'block') {
        const desiredTop = Math.max(16, (innerHeight - rect.height) / 2);
        document.querySelector('#coordinate-view').scrollBy({top: rect.top - desiredTop, behavior: 'instant'});
      } else if (!anchor.tutorialBounds) anchor.scrollIntoView({block: 'center', behavior: 'instant'});
    }
  }
  position();
}
function close(restore = true) {
  active = false;
  overlay.hidden = true;
  cancelAnimationFrame(frame);
  if (restore && trigger?.isConnected && visible(trigger)) trigger.focus({preventScroll:true});
}
function start(button) {
  close(false);
  trigger = button;
  context = currentContext();
  steps = guideSteps(context).filter(([selector]) => target(selector));
  if (!steps.length) return;
  // The focused-block view is a dialog, so its tutorial must live inside it.
  (context === 'block' ? document.querySelector('#coordinate-view') : document.body).append(overlay);
  index = 0;
  overlay.hidden = false;
  active = true;
  showStep();
  overlay.querySelector('[data-close-tour]').focus({preventScroll:true});
  function follow() {
    if (!active) return;
    if (currentContext() !== context || document.querySelector('dialog[open]')) {close(false); return;}
    position();
    frame = requestAnimationFrame(follow);
  }
  frame = requestAnimationFrame(follow);
}
back.onclick = () => {if (index > 0) {index--; showStep();}};
next.onclick = () => {if (index < steps.length - 1) {index++; showStep();} else close();};
overlay.querySelector('[data-close-tour]').onclick = () => close();
tryIt.onclick = () => {
  const anchor = target(lastTarget);
  close(false);
  // Focus rather than simulate a click: native selects and 3D controls remain usable.
  (anchor?.matches('details') ? anchor.querySelector('summary') : anchor)?.focus({preventScroll:true});
};
document.addEventListener('click', event => {
  const button = event.target.closest('[data-tutorial]');
  if (button) start(button);
});
document.addEventListener('keydown', event => {
  if (active && event.key === 'Escape') {
    event.preventDefault(); event.stopImmediatePropagation(); close();
  }
}, true);
window.addEventListener('pagehide', () => close(false));
