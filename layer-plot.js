// Each point is one recorded next-token prediction at the selected layer.
export function tokenDifferences(history, layer, strategy, method, throughStep) {
  return history.filter(frame => frame.step <= throughStep).map(frame => {
    const rows = frame.layer_differences || frame;
    const scores = method === 'jlens' ? rows.jlens?.[strategy] : rows.logit_lens;
    const entry = scores?.[layer];
    return {step: frame.step, difference: typeof entry === 'number' ? entry : entry?.difference ?? null};
  });
}

export function createLayerPlot(host, method, onLayer, onToken) {
  const title = method === 'jlens' ? 'Jlens verbalization' : 'Logit lens verbalization';
  host.innerHTML = `<h2>${title}</h2><label class="plot-layer-control">Layer <select aria-label="${title} plot layer"></select></label>
    <p class="plot-step">Choose a prompt to start</p>
    <svg viewBox="0 0 480 380" role="img" aria-label="Injection logit minus safe logit over response tokens"></svg>
    <p class="plot-reading">Scores before each response token. Positive favors injection; negative favors safe.</p><p class="plot-point"></p>`;
  const svg = host.querySelector('svg'), step = host.querySelector('.plot-step'), point = host.querySelector('.plot-point'), select = host.querySelector('select');
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
    delete svg.dataset.step; delete svg.dataset.differences;
    step.textContent = frame ? `${level ? level + ' · ' : ''}Response token ${frame.step}${method === 'jlens' ? ' · ' + (strategy === 'strategy1' ? 'Strategy 1' : 'Strategy 2') : ''}` : 'Choose a prompt to start';
    const all = tokenDifferences(currentHistory, selected, strategy, method, Infinity);
    const data = all.filter(row => row.step <= (frame?.step || 0));
    const known = all.filter(row => Number.isFinite(row.difference));
    const limit = known.reduce((maximum, row) => Math.max(maximum, Math.abs(row.difference)), 5) * 1.1;
    const plotHeight = Math.max(230, svg.clientHeight / Math.max(1, svg.clientWidth) * 480);
    svg.setAttribute('viewBox', `0 0 480 ${plotHeight}`);
    const center = (34 + plotHeight - 74) / 2, amplitude = (plotHeight - 108) / 2;
    const x = token => 76 + (token - 1) / Math.max(1, tokenCount - 1) * 360;
    const y = difference => center - difference / limit * amplitude;
    for (const value of [-limit, 0, limit]) {
      add('line', {x1: 76, x2: 436, y1: y(value), y2: y(value), class: value === 0 ? 'plot-zero' : 'plot-grid'});
      add('text', {x: 66, y: y(value) + 5, 'text-anchor': 'end', class: 'plot-tick'}, Number(value.toPrecision(3)).toString());
    }
    const ticks = [...new Set([1, Math.max(1, Math.round(tokenCount / 2)), Math.max(1, tokenCount)])];
    ticks.forEach(token => add('text', {x: x(token), y: plotHeight - 48, 'text-anchor': 'middle', class: 'plot-tick'}, String(token)));
    add('text', {x: 256, y: plotHeight - 14, 'text-anchor': 'middle', class: 'plot-axis'}, 'Response token');
    add('text', {x: 19, y: center, transform: `rotate(-90 19 ${center})`, 'text-anchor': 'middle', class: 'plot-axis'}, 'Injection logit minus safe logit');
    point.textContent = '';
    if (!frame) return;
    let connected = false;
    const path = data.map(row => {
      if (!Number.isFinite(row.difference)) { connected = false; return ''; }
      const command = `${connected ? 'L' : 'M'}${x(row.step)},${y(row.difference)}`;
      connected = true; return command;
    }).join(' ');
    add('path', {d: path, class: 'plot-line'});
    const rows = method === 'jlens' ? frame.jlens[strategy] : frame.logit_lens;
    const value = rows[selected].difference;
    add('line', {x1: x(frame.step), x2: x(frame.step), y1: 34, y2: plotHeight - 74, class: 'plot-cursor'});
    add('circle', {cx: x(frame.step), cy: y(value), r: 5, class: 'plot-current'});
    point.textContent = `Layer ${layers[selected]} · Logit difference: ${Number(value.toPrecision(5))}`;
    svg.dataset.step = String(frame.step);
    svg.dataset.layer = String(layers[selected]);
    svg.dataset.strategy = strategy;
    svg.dataset.differences = JSON.stringify(data.map(row => row.difference));
  }
  update(null, [], 'strategy1', 0);
  return {update};
}
