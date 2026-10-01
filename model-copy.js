// Descriptions use the selected model's saved configuration, not Antares defaults.
export function modelSummary(model) {
  const linear = model.layerTypes.filter(type => type === 'linear_attention').length;
  const full = model.layers - linear;
  const hidden = model.hidden.toLocaleString(), mlp = model.mlp.toLocaleString();
  const attention = linear
    ? `${linear} linear attention layers · ${full} full attention layers`
    : `${model.heads} attention heads per layer`;
  return {
    size: `${model.layers} layers · ${hidden} hidden state values per token`,
    detail: `${attention} · MLP width: ${mlp} hidden units`,
    architecture: `Each token has ${hidden} hidden state values. The MLP (multilayer perceptron) expands this to ${mlp} internal activations, uses SiLU (a smooth activation) and a learned gate, and projects back to ${hidden} output values.`,
    attention: linear
      ? `The text model has ${linear} linear attention layers and ${full} full attention layers. Full attention uses ${model.heads} attention heads (${model.headDim} values per head) sharing ${model.kvHeads} key/value groups (shared token representations); linear attention updates a running memory of earlier tokens. The vision encoder is not used for these prompts.`
      : `Each of the ${model.layers} layers uses ${model.heads} attention heads sharing ${model.kvHeads} key/value groups (shared token representations). The displayed attention numbers are the output vector after combining information from the tokens read so far.`,
    normalization: `RMS normalization divides the vector by its root mean square, with ${model.rmsNormEps} added before taking the square root to avoid division by zero. It then multiplies by ${model.normWeightOffset === 1 ? 'one plus each learned weight' : 'learned weights'}.`,
    scaling: model.residualMultiplier === 1
      ? 'Attention and MLP outputs are added back to the hidden state without extra scaling. Token scores also have no extra division before softmax.'
      : `Each attention and MLP output is multiplied by ${model.residualMultiplier} before being added back to the hidden state. The final token scores are divided by ${model.logitsScaling} before softmax. Both lenses use that same token-score division.`,
    vocabulary: `Softmax compares ${model.vocabSize.toLocaleString()} vocabulary tokens. ${model.tiedEmbeddings ? 'The same weights are used to embed input tokens and to score output tokens.' : 'The input embedding matrix and output unembedding matrix are separate learned matrices.'}`,
    layers: `The stacks show ${model.recordedLayers.length} recorded layers, numbered ${model.recordedLayers[0]} to ${model.recordedLayers.at(-1)}. The model’s final layer is layer ${model.layers - 1}. Its actual next-token scores are under Full response.`
  };
}
