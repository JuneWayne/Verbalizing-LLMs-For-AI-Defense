"""Reuse the fixed selections and their existing safe responses without generating new data"""

from pathlib import Path
import argparse
import hashlib
import json

from transformers import AutoTokenizer
import pandas as pd

from scripts.curation.common import ROOT, digest, messages, read_jsonl, write_json, write_jsonl
from scripts.curation.evaluation.evaluate import testing_pairs
from scripts.curation.training.run_manifest import read_config
from scripts.curation.training.training_helpers import check_rows, check_sources, check_splits, inspect_data


def prepare(config_path):
    config = read_config(config_path)
    source = ROOT / config['selection_source']
    original = source.parent
    destination = ROOT / config['data']
    # Preserve existing prepared pairs instead of replacing a previous experiment
    # A new data folder can be chosen in the configuration for a separate run
    if (destination / 'validation.jsonl').exists() or list((destination / 'training').glob('*.jsonl')):
        raise FileExistsError('Prepared data already exists; use --check-only or choose a new data folder')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model'], local_files_only=True)
    vocabulary = json.loads((ROOT / config['model'] / 'config.json').read_text())['vocab_size']
    groups, review, too_long = {}, [], []
    source_hashes = {}

    # prepare all eight selected groups and their shared validation responses
    for method in [*config['methods'], 'validation']:
        if method == 'validation':
            token_path = original / 'lora_validation.jsonl'
            original_rows = read_jsonl(ROOT / config['validation_source'])
            metadata = json.loads(token_path.with_suffix('.json').read_text())
            expected_hash = metadata['sha256']
        else:
            token_path = original / 'lora_training_odile' / f'{method}.jsonl'
            original_rows = read_jsonl(source / method / 'pairs.jsonl')
            metadata = json.loads(token_path.with_suffix('.json').read_text())
            expected_hash = metadata['training_sha256']
            if digest(source / method / 'pairs.jsonl') != metadata['selection_sha256']:
                raise ValueError(f'The original selection changed: {method}')
        if digest(token_path) != expected_hash:
            raise ValueError(f'The original response tokens changed: {method}')
        source_hashes[str(token_path.relative_to(ROOT))] = expected_hash
        originals = {row['pair_id']: row for row in original_rows}
        rows = []
        for old in read_jsonl(token_path):
            prompt = originals[old['pair_id']]

            # Verify that the response tokens belong to the configured task and system prompt
            for condition in ('safe', 'injection'):
                prefix = tokenizer.apply_chat_template(messages(prompt, condition, config), tokenize=True, add_generation_prompt=True, return_dict=False)
                if prefix != old['prefix_ids'][condition]:
                    raise ValueError(f'Prompt formatting differs: {method}, pair {old["pair_id"]}')

            # the final training target is the existing safe response, including its natural end token
            target = list(old['completion_ids']['safe'])
            if not target or target[-1] != tokenizer.eos_token_id:
                raise ValueError('The existing safe response has no natural end token')

            # preserve the complete safe response, including its original end token
            row = {'pair_id': old['pair_id'], 'source_id': str(prompt['user_prompt_id']), 'clean_data_sha256': hashlib.sha256(prompt['clean_data'].encode()).hexdigest(), 'prefix_ids': old['prefix_ids'], 'completion_ids': old['completion_ids'], 'task_response_ids': target}
            rows.append(row)
            if len(target) > config['max_task_response_tokens']:
                too_long.append({'method': method, 'pair_id': row['pair_id'], 'tokens': len(target)})
            review.append({'method': method, 'pair_id': row['pair_id'], 'user_instruction': prompt['user_instruction'], 'safe_prompt': prompt['clean_data'], 'injection_prompt': prompt['injected_data'], 'task_response': tokenizer.decode(target, skip_special_tokens=True), 'task_response_tokens': len(target), 'target_source': 'existing safe response', 'correctness_independently_verified': False})
        check_sources(rows, original_rows, exact=method != 'validation')
        groups[method] = rows

    # compare each training group with the same validation set before exporting prepared inputs
    for method in config['methods']:
        check_splits(groups[method], groups['validation'])

    # Apply token and pair-count checks to every group before writing training inputs
    # Overlength responses require an explicit setting change rather than silent truncation
    for method, rows in groups.items():
        count = config['validation_pairs'] if method == 'validation' else config['training_pairs']
        check_rows(rows, count, config, tokenizer.eos_token_id, vocabulary)

    # Choose the fixed testing subset before training if it has not been prepared yet
    # Existing testing records remain unchanged when repeating a run
    testing_path = ROOT / config['testing_source']
    if not testing_path.exists():
        write_jsonl(testing_path, testing_pairs(config['testing_pairs']))

    # Export each complete reference response with its original prompt prefixes
    for method, rows in groups.items():
        path = destination / 'validation.jsonl' if method == 'validation' else destination / 'training' / f'{method}.jsonl'
        write_jsonl(path, rows)
    pd.DataFrame(review).to_csv(destination / 'targets_for_review.csv', index=False)
    # Recheck prompt tokenization and separation from testing using the exported records
    inspect_data(config)
    report = {'pair_counts': {method: len(rows) for method, rows in groups.items()}, 'maximum_task_response_tokens': max(row['task_response_tokens'] for row in review), 'overlength_targets': too_long, 'source_hashes': source_hashes, 'training_started': False, 'target_policy': 'Existing model-generated safe responses; task correctness has not been independently verified'}
    write_json(destination / 'preparation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/curation/configs/config_reference_groups.json')
    parser.add_argument('--check-only', action='store_true', help='Check existing prepared pairs without replacing them or training')
    args = parser.parse_args()
    path = args.config.resolve()
    if args.check_only:
        config = read_config(path)
        groups, validation, testing = inspect_data(config)
        print(json.dumps({'pair_counts': {method: len(rows) for method, rows in groups.items()}, 'validation_pairs': len(validation), 'testing_pairs': len(testing), 'training_started': False}, indent=2))
    else:
        print(json.dumps(prepare(path), indent=2))
