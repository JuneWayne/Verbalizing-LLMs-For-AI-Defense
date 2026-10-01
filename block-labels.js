// Keep component names identical in the layer and its focused view.
export const blockTitles = {
  attention: 'Attention output', mlp: 'MLP output', hidden: 'Hidden state', jacobian: 'Jacobian matrix',
  transformed: 'Estimated final hidden state', normalized: 'Normalized hidden state', unembedding: 'Unembedding matrix',
  logits: 'Logits', softmax: 'Token probability distribution'
};

export const blockDescriptions = {
  attention: 'Attention combines information from tokens already read. These numbers show the resulting update to the hidden state.',
  mlp: 'The MLP is a small neural network inside each layer. It expands the hidden state, transforms it, then projects back to the original size. These numbers are the output values added back to the hidden state.',
  hidden: 'This hidden state is the residual stream vector at the selected layer and token position. The residual stream carries the representation that attention and MLP blocks update as it passes through the model. Both lenses read this vector to predict the next token.',
  jacobian: 'A Jacobian matrix describes how small changes to hidden state values affect the final-layer hidden state. Each row corresponds to one final hidden state value, and each column to one value at the selected layer. Jlens averages these rates of change over fitting prompts, then multiplies the resulting matrix by the current hidden state to estimate the final hidden state before normalization.',
  transformed: 'Jlens estimates the final hidden state by multiplying the selected layer’s hidden state by its fitted Jacobian matrix. The estimate then passes through the model’s final RMSNorm and unembedding matrix to predict tokens.',
  normalized: 'RMSNorm (root mean square normalization) rescales the hidden state using its root mean square and learned weights. This block shows the resulting normalized hidden state. Both lenses apply the model’s final RMSNorm before the unembedding matrix.',
  unembedding: 'The unembedding matrix maps the normalized hidden state to vocabulary logits. Each displayed row contains the learned weights for one token. Multiply each weight in that row by the normalized hidden state value in the matching column, then sum those products to calculate that token’s logit.',
  logits: 'Each token receives a score, called a logit. Higher scores mean stronger next-token predictions. Softmax converts these scores into probabilities.',
  softmax: 'Softmax converts scores into a probability distribution over the full set of vocabulary tokens that the model is trained on. Their probability scores add up to 100% in total. Here the visualization is only showing you the top 5 token probability scores.'
};

export const vectorValueNames = {
  hidden: 'Hidden state value', attention: 'Attention output value', mlp: 'MLP output value',
  transformed: 'Estimated final hidden state value', normalized: 'Normalized hidden state value',
  logits: 'Logit', softmax: 'Token probability'
};

export function describeBlock(key, model, layer) {
  if (key === 'attention' && model?.layerTypes[layer] === 'linear_attention') {
    return 'This linear attention layer updates a running memory of earlier tokens. Its output vector carries that combined information into the layer’s hidden state.';
  }
  const description = blockDescriptions[key];
  if (key === 'unembedding' && model) return description + (model.logitsScaling === 1 ? ' This model uses the summed score directly.' : ` This model then divides the score by ${model.logitsScaling} before softmax.`);
  if (['attention', 'mlp'].includes(key) && model?.residualMultiplier !== undefined && model.residualMultiplier !== 1) return description + ` This model multiplies the output by ${model.residualMultiplier} before adding it back to the hidden state.`;
  return description;
}

export function formatProbability(probability) {
  const percent = probability * 100;
  // Keep tiny nonzero probabilities visible instead of rounding them to 0%.
  return (percent > 0 && percent < 0.001 ? percent.toExponential(4) : Number(percent.toPrecision(6)).toString()) + '%';
}

export function tokenProbabilityLabel(token, probability) {
  return `${JSON.stringify(token)} probability = ${formatProbability(probability)}`;
}

export const blockTrainingDescriptions = {
  jacobian: "To fit the lens, we first run each training prompt through the model and record its hidden states. We then use automatic differentiation to calculate how a small change in an earlier layer’s hidden state would affect the final layer’s hidden state. These rates of change form a Jacobian matrix.\n\nBecause a token can influence both its own representation and later tokens, the fitter adds these effects across the selected token positions. It then averages the results across input positions and training prompts, producing one representative matrix for each inspected layer. This calculation uses positions after the first 16 tokens and before the final token, while keeping the language model’s weights unchanged.\n\nWhen inspecting a new prompt, Jlens multiplies that layer’s hidden state by its saved matrix to estimate the final hidden state."
};
