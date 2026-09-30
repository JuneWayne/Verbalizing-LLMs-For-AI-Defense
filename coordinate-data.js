import { blockTitles, vectorValueNames, formatProbability } from './block-labels.js?v=plain-math-29';
// Calculations use saved values only. A displayed window is never treated as a full sum.
export function coordinateData(frame, weights, strategy, layer, method, key) {
  const row = method === 'jlens' ? frame.jlens[strategy][layer] : frame.logit_lens[layer];
  const state = frame.layer_values[layer];
  const size = weights.hidden_size;
  const fmt = value => Number(value).toPrecision(6);
  if (key === 'jacobian') {
    return {
      title: blockTitles.jacobian, inputTitle: blockTitles.hidden, outputTitle: blockTitles.transformed,
      values: weights.jacobians[strategy][String(layer)], input: state.hidden, output: row.transformed,
      extent: `${weights.window} × ${weights.window} saved matrix values from a ${size.toLocaleString()} × ${size.toLocaleString()} matrix. Indices start at 0.`,
      explain(i, j) {
        const weight = this.values[i][j], input = this.input[j];
        return { value: weight, input, output: this.output[i], contribution: weight * input,
          equation: `${fmt(weight)} × ${fmt(input)} ≈ ${fmt(weight * input)}`,
          meaning: `Matrix value [${i}, ${j}] multiplies hidden state value ${j}. Their product, ${fmt(weight * input)}, is one number in the sum for estimated hidden state value ${i}.`,
          total: `Row ${i} contains ${size.toLocaleString()} matrix values. Multiply each by the matching hidden state value and add all ${size.toLocaleString()} multiplication results. This estimates hidden state value ${i}: ${fmt(this.output[i])}.` };
      }
    };
  }
  if (key === 'unembedding') {
    return {
      title: blockTitles.unembedding, inputTitle: blockTitles.normalized, outputTitle: blockTitles.logits,
      values: row.tokens.map(token => weights.unembedding[String(token.id)]),
      input: row.normalized, output: row.tokens.map(token => token.logit),
      names: row.tokens.map(token => token.text),
      extent: `Five token rows and ${weights.window} saved matrix values per row. Each full row has ${size.toLocaleString()} weights.`,
      explain(i, j) {
        const weight = this.values[i][j], input = this.input[j], contribution = weight * input / weights.logits_scaling;
        return { value: weight, input, output: this.output[i], contribution,
          equation: `${fmt(weight)} × ${fmt(input)}${weights.logits_scaling === 1 ? '' : ' ÷ ' + weights.logits_scaling} ≈ ${fmt(contribution)}`,
          meaning: `For ${JSON.stringify(this.names[i])}, multiply normalized hidden state value ${j} by the vocabulary weight in column ${j}${weights.logits_scaling === 1 ? '' : ', then divide by ' + weights.logits_scaling}. The result, ${fmt(contribution)}, is one of the numbers added to calculate this token’s score.`,
          total: `The normalized hidden state has ${size.toLocaleString()} values. Multiply each by its matching weight in this token’s row and add all ${size.toLocaleString()} results${weights.logits_scaling === 1 ? '' : ', then divide by ' + weights.logits_scaling}. The saved calculation gives a token score of ${fmt(this.output[i])}.` };
      }
    };
  }
  const vectors = { ...state, transformed: row.transformed, normalized: row.normalized,
    logits: row.tokens.map(token => token.logit), softmax: row.tokens.map(token => token.probability) };
  const titles = blockTitles;
  const descriptions = {
    hidden: 'One hidden state value after the layer adds its attention and MLP updates.',
    attention: 'This value is part of the attention output vector, which carries combined information from the tokens processed so far.',
    mlp: '',
    transformed: 'One estimated hidden state value from the Jacobian applied to the full hidden state vector.',
    normalized: '',
    logits: 'A vocabulary score. A larger score makes this token more likely.',
    softmax: 'The probability assigned to this token in the full vocabulary distribution.'
  };
  return {
    title: titles[key], values: vectors[key].map(value => [value]),
    names: ['logits', 'softmax'].includes(key) ? row.tokens.map(token => token.text) : null,
    extent: ['logits', 'softmax'].includes(key) ? 'Top five tokens from the full vocabulary.' : `${weights.window} saved vector values from a ${size.toLocaleString()}-dimensional vector. Indices start at 0.`,
    explain(i) {
      const value = this.values[i][0];
      return { value,
        equation: key === 'softmax' ? `exp(${fmt(row.tokens[i].logit)} − ${fmt(row.log_normalizer)}) ≈ ${fmt(value)} = ${formatProbability(value)}` : `${this.names ? JSON.stringify(this.names[i]) : vectorValueNames[key] + ' ' + i} = ${fmt(value)}${key === 'mlp' ? ' → after projecting the internal activations back to the hidden state size.' : ''}`,
        meaning: key === 'softmax' ? `${method === 'jlens' ? 'Jlens' : 'Logit lens'} assigns ${JSON.stringify(this.names[i])} a ${formatProbability(value)} probability as the next token at this layer. This lens prediction describes which token follows the text processed so far.` : descriptions[key],
        total: key === 'softmax' ? `${fmt(row.log_normalizer)} is the log of the summed exponentials of all ${weights.vocab_size.toLocaleString()} token scores. Subtracting it before exp produces this token’s share of the full distribution.` : '' };
    }
  };
}
