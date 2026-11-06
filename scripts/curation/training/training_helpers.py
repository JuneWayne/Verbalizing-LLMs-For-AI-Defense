"""Share the training schedule and prepared-data checks for reference-response LoRA"""

import hashlib
import json
import math
import random

from transformers import AutoTokenizer

from scripts.curation.common import ROOT, messages, read_jsonl


# Use the same seeded pair ordering for every selected group
# The epoch changes the shuffle seed while keeping the number of updates the same
def epoch_order(config, epoch):
    # every adapter sees its own pairs in the same seeded order for a fair comparison
    order = list(range(config['training_pairs']))

    # adding the epoch number changes the order between epochs while keeping the run repeatable
    random.Random(config['seed'] + epoch).shuffle(order)
    return order


# Use linear warmup followed by cosine decay across the complete training run
# The update parameter counts optimizer steps across epochs, so the schedule never restarts each epoch
def learning_rate(update, config):
    # spread warmup and decay across all five epochs, rather than restarting each epoch
    total = config['training_pairs'] * config['epochs']
    warmup = max(1, math.ceil(total * config['warmup_fraction']))
    if not 1 <= update <= total:
        raise ValueError('Update lies outside the approved training schedule')

    # warmup increases the rate linearly from a small value to the configured peak
    if update <= warmup:
        return config['learning_rate'] * update / warmup

    # after warmup, progress runs from 0 toward 1 across the remaining optimizer updates
    # the cosine term decreases the rate smoothly toward the configured final fraction
    progress = (update - warmup) / (total - warmup)
    floor = config['final_learning_rate_fraction']
    return config['learning_rate'] * (floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2)


# Check the selected pair count, token IDs, prompt lengths and complete reference answers
# These checks reject changed training inputs before any model weights can be updated
def check_rows(rows, count, config, eos_id, vocabulary_size):
    # reject missing or overlength targets instead of silently changing the curated data
    if len(rows) != count or len({row['pair_id'] for row in rows}) != count:
        raise ValueError('The prepared pair count or pair uniqueness differs')
    for row in rows:
        if not row.get('source_id') or not row.get('clean_data_sha256'):
            raise ValueError('Each pair needs source_id and clean_data_sha256 for split checks')

        # check both original prompt prefixes and their paired completion token sequences
        sequences = []
        for condition in ('safe', 'injection'):
            prefix = row['prefix_ids'][condition]
            completion = row['completion_ids'][condition]
            if not prefix or len(prefix) > config['max_input_tokens'] or not completion:
                raise ValueError(f'Invalid prompt or response tokens for pair {row["pair_id"]}')
            sequences.extend([prefix, completion])

        # the safe reference must include text followed by the model's natural end token
        target = row['task_response_ids']
        if len(target) < 2 or len(target) > config['max_task_response_tokens'] or target[-1] != eos_id:
            raise ValueError('Each task response must contain text and its end token, within the approved limit')
        sequences.append(target)

        # every token must be an integer index inside the model's vocabulary
        if any(type(token) is not int or not 0 <= token < vocabulary_size for ids in sequences for token in ids):
            raise ValueError('A token ID lies outside the model vocabulary')


# Keep training and validation separate by pair ID, original source ID and clean prompt content
# This catches repeated source content even when its pair ID differs
def check_splits(training, validation):
    # separate both source records and repeated clean prompts, not just pair IDs
    for field in ('pair_id', 'source_id', 'clean_data_sha256'):
        if {row[field] for row in training} & {row[field] for row in validation}:
            raise ValueError(f'Training and validation overlap on {field}')


# Match prepared tokenized pairs to the original selected FIPI records
# The source ID and clean-content hash identify the original task behind each injection variant
def check_sources(rows, sources, exact=False):
    # source_id refers to the original FIPI user_prompt_id, shared by injection variants
    source = {row['pair_id']: row for row in sources}
    if exact and {row['pair_id'] for row in rows} != set(source):
        raise ValueError('Prepared training pairs differ from the existing selection')

    # compare prepared metadata with the original selected pair, including its clean-content hash
    for row in rows:
        original = source.get(row['pair_id'])
        if original is None or row['source_id'] != str(original['user_prompt_id']) or row['clean_data_sha256'] != hashlib.sha256(original['clean_data'].encode()).hexdigest():
            raise ValueError('Prepared source metadata differs from the original FIPI pair')


# Confirm that the tokenized prompt exactly matches the training prefix for each condition
# The same tokenizer and prompt format must be used when preparing and checking a reference answer
def check_prompt_tokens(rows, prepared, tokenizer, config):
    # the response must start after the exact prompt used during training
    prefixes = {row['pair_id']: row['prefix_ids'] for row in prepared}
    for row in rows:
        for condition in ('safe', 'injection'):
            # reconstruct the exact chat-formatted prefix for this safe or injection condition
            text = tokenizer.apply_chat_template(messages(row, condition, config), tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(text, add_special_tokens=False)

            # reject any changed wording, chat-template markers or tokenization before training
            if ids != prefixes[row['pair_id']][condition]:
                raise ValueError(f'The {condition} prompt differs from training for {row["pair_id"]}')
            if len(ids) > config['max_input_tokens']:
                raise ValueError('A full prompt exceeds the input limit; truncation is disabled')


# Inspect every training group and shared validation/testing split before training starts
# Shared checks validate token lengths, original source records and exact prompt prefixes
# Testing pairs must remain separate from both training and validation by source and content
def inspect_data(config):
    # check token lengths and split membership before any adapter weights can change

    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model'], local_files_only=True)
    vocabulary = json.loads((ROOT / config['model'] / 'config.json').read_text())['vocab_size']
    data = ROOT / config['data']
    validation = read_jsonl(data / 'validation.jsonl')
    testing = read_jsonl(ROOT / config['testing_source'])
    check_rows(validation, config['validation_pairs'], config, tokenizer.eos_token_id, vocabulary)

    # compare validation token records with the source prompts from which they were prepared
    validation_source = read_jsonl(ROOT / config['validation_source'])
    check_sources(validation, validation_source)
    validation_ids = {row['pair_id'] for row in validation}
    check_prompt_tokens([row for row in validation_source if row['pair_id'] in validation_ids], validation, tokenizer, config)
    if len(testing) != config['testing_pairs'] or len({row['pair_id'] for row in testing}) != len(testing):
        raise ValueError('Testing pair count changed')
    groups = {}

    # inspect every selected group using the same token limits and source-matching checks
    for method in config['methods']:
        training = read_jsonl(data / 'training' / f'{method}.jsonl')
        source = read_jsonl(ROOT / config['selection_source'] / method / 'pairs.jsonl')
        check_rows(training, config['training_pairs'], config, tokenizer.eos_token_id, vocabulary)
        check_sources(training, source, exact=True)
        check_prompt_tokens(source, training, tokenizer, config)
        check_splits(training, validation)
        groups[method] = training

    # all eight groups use the same testing pairs, separate by source and clean prompt
    for rows in [validation, *groups.values()]:
        for field, test_values in (
            ('pair_id', {row['pair_id'] for row in testing}),
            ('source_id', {str(row['user_prompt_id']) for row in testing}),
            ('clean_data_sha256', {hashlib.sha256(row['clean_data'].encode()).hexdigest() for row in testing}),
        ):
            if {row[field] for row in rows} & test_values:
                raise ValueError(f'Testing overlaps training or validation on {field}')

    # measure complete testing prompts so no task or injection is silently truncated
    for row in testing:
        for condition in ('safe', 'injection'):
            ids = tokenizer.apply_chat_template(messages(row, condition, config), tokenize=True, add_generation_prompt=True)
            if len(ids) > config['max_input_tokens']:
                raise ValueError('A testing prompt exceeds the approved input limit')
    return groups, validation, testing
