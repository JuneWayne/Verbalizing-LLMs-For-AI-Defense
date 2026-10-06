// Each point is one recorded next-token prediction at the selected layer.
export function tokenDifferences(history, layer, strategy, method, throughStep) {
  return history.filter(frame => frame.step <= throughStep).map(frame => {
    const rows = frame.layer_differences || frame;
    const scores = method === 'jlens' ? rows.jlens?.[strategy] : rows.logit_lens;
    const entry = scores?.[layer];
    return {step: frame.step, difference: typeof entry === 'number' ? entry : entry?.difference ?? null};
  });
}

export function tokenLogits(history, layer, strategy, method, throughStep) {
  return history.filter(frame => frame.step <= throughStep).map(frame => {
    const rows = frame.layer_logits || frame;
    const entry = (method === 'jlens' ? rows.jlens?.[strategy] : rows.logit_lens)?.[layer];
    const scores = entry?.label_logits || entry;
    return {step: frame.step, injection: scores?.injection ?? null, safe: scores?.safe ?? null};
  });
}

const scoreText = value => Number(value.toPrecision(6)).toString();

export function plotCaption(layer, method, injection, safe) {
  const lens = method === 'jlens' ? 'Jlens' : 'Logit lens';
  if (!Number.isFinite(injection) || !Number.isFinite(safe)) return `Token scores are unavailable at layer ${layer}.`;
  if (injection === safe) return `The lines are tied. At layer ${layer}, ${lens} gives injection and safe the same next-token score: ${scoreText(safe)}.`;
  const leading = injection > safe;
  return `The ${leading ? 'orange' : 'blue'} line is higher. At layer ${layer}, ${lens} scores ${leading ? 'injection above safe' : 'safe above injection'} for the next token: ${scoreText(leading ? injection : safe)} versus ${scoreText(leading ? safe : injection)}.`;
}

export function createLayerPlot(host, method, onLayer, onToken) {
  const title = method === 'jlens' ? 'Jlens verbalization' : 'Logit lens verbalization';
  host.innerHTML = `<h2>${title}</h2><label class="plot-layer-control">Layer <select aria-label="${title} plot layer"></select></label>
    <svg viewBox="0 0 480 380" role="img" aria-label="Token logits"></svg>
    <p class="plot-caption">Choose a prompt to start</p>`;
  const svg = host.querySelector('svg'), caption = host.querySelector('.plot-caption'), select = host.querySelector('select');
  const ns = 'http://www.w3.org/2000/svg';
  select.disabled = true;
  select.onchange = () => onLayer(Number(select.value));
  const add = (name, attributes, text = '') => {
    const element = document.createElementNS(ns, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    element.textContent = text; svg.append(element); return element;
  };
  let currentHistory = [], tokenCount = 0, lastUpdate = null, resizing = false;
  const observer = new ResizeObserver(() => {
    if (!lastUpdate || resizing) return;
    resizing = true;
    requestAnimationFrame(() => { resizing = false; update(...lastUpdate); });
  });
  observer.observe(svg);
  // Clicking the trace seeks the shared recording; the playback slider also works by keyboard.
  svg.onclick = event => {
    if (!tokenCount) return;
    const bounds = svg.getBoundingClientRect();
    const token = Math.round(((event.clientX - bounds.left) * 480 / bounds.width - 76) / 360 * Math.max(1, tokenCount - 1));
    onToken(Math.max(0, Math.min(tokenCount - 1, token)));
  };
  function update(frame, layers, strategy, selected, history = [], level = '') {
    lastUpdate = [frame, layers, strategy, selected, history, level];
    if (select.options.length !== layers.length) select.replaceChildren(...layers.map((layer, index) => new Option(String(layer), index)));
    select.disabled = !frame; select.value = selected;
    currentHistory = history;
    tokenCount = currentHistory.length;
    svg.replaceChildren();
    delete svg.dataset.step; delete svg.dataset.differences; delete svg.dataset.logits;
    caption.textContent = 'Choose a prompt to start';
    const all = tokenLogits(currentHistory, selected, strategy, method, Infinity);
    const data = all.filter(row => row.step <= (frame?.step || 0));
    const known = all.flatMap(row => [row.injection, row.safe]).filter(Number.isFinite);
    const low = known.length ? Math.min(...known) : -1, high = known.length ? Math.max(...known) : 1;
    const padding = Math.max(1, (high - low) * .1), minimum = low - padding, maximum = high + padding;
    const plotHeight = Math.max(230, svg.clientHeight / Math.max(1, svg.clientWidth) * 480);
    svg.setAttribute('viewBox', `0 0 480 ${plotHeight}`);
    const center = (34 + plotHeight - 74) / 2;
    const x = token => 76 + (token - 1) / Math.max(1, tokenCount - 1) * 360;
    const y = value => 34 + (maximum - value) / (maximum - minimum) * (plotHeight - 108);
    for (const value of [minimum, (minimum + maximum) / 2, maximum]) {
      add('line', {x1: 76, x2: 436, y1: y(value), y2: y(value), class: value === 0 ? 'plot-zero' : 'plot-grid'});
      add('text', {x: 66, y: y(value) + 5, 'text-anchor': 'end', class: 'plot-tick'}, Number(value.toPrecision(3)).toString());
    }
    const ticks = [...new Set([1, Math.max(1, Math.round(tokenCount / 2)), Math.max(1, tokenCount)])];
    ticks.forEach(token => add('text', {x: x(token), y: plotHeight - 48, 'text-anchor': 'middle', class: 'plot-tick'}, String(token)));
    add('text', {x: 256, y: plotHeight - 14, 'text-anchor': 'middle', class: 'plot-axis'}, 'Response token');
    add('text', {x: 19, y: center, transform: `rotate(-90 19 ${center})`, 'text-anchor': 'middle', class: 'plot-axis'}, 'Token logit');
    if (!frame) return;
    const rows = method === 'jlens' ? frame.jlens[strategy] : frame.logit_lens;
    for (const label of ['injection', 'safe']) {
      let connected = false;
      const path = data.map(row => {
        if (!Number.isFinite(row[label])) { connected = false; return ''; }
        const command = `${connected ? 'L' : 'M'}${x(row.step)},${y(row[label])}`;
        connected = true; return command;
      }).join(' ');
      add('path', {d: path, class: `plot-line plot-${label}`, 'data-label': label});
      const value = rows[selected].label_logits[label];
      if (Number.isFinite(value)) add('circle', {cx: x(frame.step), cy: y(value), r: 4, class: `plot-current plot-${label}`});
    }
    add('line', {x1: x(frame.step), x2: x(frame.step), y1: 34, y2: plotHeight - 74, class: 'plot-cursor'});
    const scores = rows[selected].label_logits;
    caption.textContent = plotCaption(layers[selected], method, scores.injection, scores.safe);
    // Separate nearby labels while keeping a connector to each exact score.
    const ends = ['injection', 'safe'].filter(label => Number.isFinite(scores[label]))
      .map(label => ({label, position: y(scores[label])})).sort((a, b) => a.position - b.position);
    if (ends.length === 2 && ends[1].position - ends[0].position < 28) {
      const middle = (ends[0].position + ends[1].position) / 2;
      ends[0].position = middle - 14; ends[1].position = middle + 14;
    }
    for (const end of ends) {
      const labelY = Math.max(18, Math.min(plotHeight - 80, end.position));
      const text = add('text', {y: labelY + 4, class: `plot-end-label plot-${end.label}`, 'data-label': end.label}, `${end.label} · ${scoreText(scores[end.label])}`);
      const width = text.getComputedTextLength();
      const labelX = x(frame.step) + width + 20 <= 472 ? x(frame.step) + 12 : x(frame.step) - width - 12;
      text.setAttribute('x', labelX);
      const connector = add('line', {x1: x(frame.step), y1: y(scores[end.label]), x2: labelX < x(frame.step) ? labelX + width + 4 : labelX - 4, y2: labelY, class: `plot-label-link plot-${end.label}`});
      svg.insertBefore(connector, text);
    }
    svg.dataset.step = String(frame.step);
    svg.dataset.layer = String(layers[selected]);
    svg.dataset.strategy = strategy;
    svg.dataset.differences = JSON.stringify(tokenDifferences(currentHistory, selected, strategy, method, frame.step).map(row => row.difference));
    svg.dataset.logits = JSON.stringify(data);
  }
  update(null, [], 'strategy1', 0);
  return {update};
}
