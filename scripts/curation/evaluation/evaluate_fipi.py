"""Generate and summarize the fixed FIPI testing responses after checkpoint selection"""

import csv

import torch

from scripts.curation.common import write_json, write_jsonl
from scripts.curation.evaluation.evaluate import repetition_check, response_record
from scripts.curation.preparation.Record_hidden_state_vectors_and_model_response import record_batch


# Evaluate the base model or one validation-selected adapter on the same testing pairs
# Each pair receives a safe response and an injection response under greedy generation
# The evaluator retains generated tokens, end-token completion and the requested injected-answer check
# Response records and repetition flags preserve the scoring used in the original experiment
# The model and tokenizer parameters are the loaded language model and its text encoder
# The original dictionary supplies the prompt settings, while the plan dictionary supplies the response limit
# The groups dictionary maps each split name to its prompt pairs, and name identifies the evaluated checkpoint
# The folder parameter sets the experiment directory used for the response records and review table
def evaluate(model, tokenizer, original, plan, groups, folder, name):

    # Keep each checkpoint's responses together and refuse to replace an existing evaluation
    destination = folder / 'evaluation' / name
    destination.mkdir(parents=True, exist_ok=False)

    # Disable training behavior such as dropout so repeated tests use the same generation settings
    # Cache earlier attention calculations to avoid computing them again for every response token
    model.eval()
    model.config.use_cache = True

    # An end token marks a naturally completed response
    # Some models allow several end tokens, so keep all accepted token IDs in one set
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    # The tokens variable limits newly generated response tokens using the approved evaluation plan
    tokens = plan['evaluation_max_new_tokens']
    responses, summaries = [], []

    # Generate both versions of every pair so the task can be compared with and without the injection
    # Each row contains the shared user instruction, safe content and content with an inserted injection
    for split, rows in groups.items():
        for row in rows:
            for condition in ('safe', 'injection'):

                # record_batch builds the model input and generates one response with greedy token selection
                # Keep the original prompt settings and replace only the response token limit
                # The returned list contains one response, so index 0 retrieves its generated token IDs
                ids = record_batch(model, tokenizer, {**original, 'max_new_tokens': tokens}, [(row, condition)], generate=True)[0]

                # Decode the response and check whether it reached an end token
                # The witness_present field checks for the answer phrase requested by the injection
                # The dataset stores that requested phrase in its witness field
                result = response_record(row, condition, ids, tokenizer, eos, tokens)

                # Check the final 512 tokens for repeated sequences of eight tokens
                # Flag repetition when over 80% of those sequences repeat, at least 128 tokens were inspected
                # and the response reached its token limit without an end token
                result.update(repetition_check(result, tokens))

                # Attach the split and checkpoint names so later comparisons can identify each response
                result.update(split=split, checkpoint=name)
                responses.append(result)

                # Update the response file after every prompt so completed responses survive an interruption
                # Each pair contributes two prompts to the total, one safe and one injection
                write_jsonl(destination / 'responses.jsonl', responses)
                write_json(folder / 'progress.json', {'stage': 'evaluation', 'checkpoint': name, 'completed_prompts': len(responses), 'total_prompts': 2 * sum(map(len, groups.values()))})
                print(f'{name}: generated {len(responses)} prompts', flush=True)

        # Summarize safe and injection responses separately within the current split
        # Adding Boolean flags counts responses with repetition, unfinished generation or the requested phrase
        for condition in ('safe', 'injection'):
            subset = [row for row in responses if row['split'] == split and row['condition'] == condition]
            summaries.append({
                'checkpoint': name,
                'split': split,
                'condition': condition,
                'prompts': len(subset),
                'repetitive_at_limit': sum(row['repetitive_at_limit'] for row in subset),
                'unfinished': sum(not row['finished'] for row in subset),
                'requested_phrase_present': sum(row['witness_present'] for row in subset),
                # Leave reviewed task accuracy and attack success blank until the responses are assessed
                'task_completion_accuracy': None,
                'verified_injection_success_rate': None,
            })

    write_json(destination / 'summary.json', summaries)

    # Match each generated response back to its input pair using the pair_id dictionary field
    # The review table puts the user instruction, prompt and model response beside each other
    lookup = {row['pair_id']: row for rows in groups.values() for row in rows}
    fields = ['split', 'pair_id', 'condition', 'user_instruction', 'prompt', 'injection_requested_phrase', 'response', 'finished', 'repetitive_at_limit', 'task_completed_correctly', 'injection_followed']
    with (destination / 'responses_for_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for response in responses:
            row = lookup[response['pair_id']]

            # Use clean_data for a safe response and injected_data for an injection response
            # Leave task_completed_correctly and injection_followed empty for response review
            writer.writerow({
                **{key: response[key] for key in ('split', 'pair_id', 'condition', 'response', 'finished', 'repetitive_at_limit')},
                'user_instruction': row['user_instruction'],
                'prompt': row['clean_data' if response['condition'] == 'safe' else 'injected_data'],
                'injection_requested_phrase': row['witness'],
            })

    # Disable generation caching before control returns to the training workflow
    # Release unused GPU memory held by PyTorch's allocator
    model.config.use_cache = False
    torch.cuda.empty_cache()
