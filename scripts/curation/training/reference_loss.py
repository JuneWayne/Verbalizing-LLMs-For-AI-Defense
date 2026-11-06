"""Calculate the reference-response objective used by the final LoRA experiment"""

import torch
import torch.nn.functional as F


# Train the injection prompt toward the paired safe response by predicting its reference tokens
# Each reference token is scored after the injection input and the earlier reference tokens
# The target includes the natural end token, allowing training to learn when the response should finish
# Cross-entropy averages the negative log probability across all reference tokens
def task_response_loss(model, row, device):
    # the last prompt token predicts the first answer token, including the natural end token
    # prefix contains the injection prompt tokens and target contains its safe reference response
    # placing them together lets one forward pass score every reference token using teacher forcing
    prefix, target = row['prefix_ids']['injection'], row['task_response_ids']
    tokens = torch.tensor([prefix + target], device=device)
    result = model(input_ids=tokens, use_cache=False, logits_to_keep=len(target) + 1)

    # keep the scores that predict the reference tokens, beginning at the last prompt position
    # the final returned position predicts a token after the reference, so it is excluded
    scores = result.logits[0, :-1].float()

    # the target token IDs identify the expected word or token at each position
    # cross-entropy averages their negative log probabilities, including the end token
    return F.cross_entropy(scores, torch.tensor(target, device=device))
