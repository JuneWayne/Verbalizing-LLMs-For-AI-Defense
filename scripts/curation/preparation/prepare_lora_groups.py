"""Reuse the fixed selections and their existing safe responses without generating new data"""

from pathlib import Path
import argparse
import hashlib
import json
import time

from transformers import AutoTokenizer
import pandas as pd

from scripts.curation.common import METHODS, ROOT, digest, messages, read_jsonl, write_json, write_jsonl
from scripts.curation.training.data_approval import TRAINING_CODE_FILES, approved_inputs
from scripts.curation.training.training_helpers import check_rows, check_sources, check_splits


def prepare(config_path, record_approval=False):
    config = json.loads(config_path.read_text())
    source = ROOT / config['selection_source']
    original = source.parent
    destination = ROOT / config['data']
    if (destination / 'approval.json').exists():
        raise FileExistsError('Approved data already exists; preserve the approved version')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model'], local_files_only=True)
    vocabulary = json.loads((ROOT / config['model'] / 'config.json').read_text())['vocab_size']
    groups, review, too_long = {}, [], []
    source_hashes = {}

    # prepare all eight selected groups and their shared validation responses
    for method in [*METHODS, 'validation']:
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

            # verify that the original tokens used the approved neutral system prompt
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
    for method in METHODS:
        check_splits(groups[method], groups['validation'])

    # draft exports keep every pair and do not truncate overlength responses
    for method, rows in groups.items():
        path = destination / 'validation.jsonl' if method == 'validation' else destination / 'training' / f'{method}.jsonl'
        write_jsonl(path, rows)
    pd.DataFrame(review).to_csv(destination / 'targets_for_review.csv', index=False)
    report = {'pair_counts': {method: len(rows) for method, rows in groups.items()}, 'maximum_task_response_tokens': max(row['task_response_tokens'] for row in review), 'overlength_targets': too_long, 'source_hashes': source_hashes, 'training_started': False, 'target_policy': 'Existing safe responses accepted by the user; no independent correctness assessment'}
    write_json(destination / 'preparation.json', report)

    # ordinary preparation returns the data report without authorizing a training run
    if not record_approval:
        return report

    if config['status'] != 'approved':
        raise ValueError('The exact configuration must be approved before recording authorization')
    for method, rows in groups.items():
        count = config['validation_pairs'] if method == 'validation' else config['training_pairs']
        check_rows(rows, count, config, tokenizer.eos_token_id, vocabulary)
    required = [destination / 'validation.jsonl', destination / 'targets_for_review.csv', destination / 'preparation.json', ROOT / config['model_hashes'], ROOT / config['validation_source']]
    required.extend(destination / 'training' / f'{method}.jsonl' for method in METHODS)
    required.extend(source / method / 'pairs.jsonl' for method in METHODS)
    required.extend(ROOT / name for name in TRAINING_CODE_FILES)
    required.append(Path(__file__))
    required.append(ROOT / 'bash/train_reference_groups.sbatch')
    approval = {'training_authorized': True, 'targets_accepted': True, 'correctness_independently_verified': False, 'execution': config['execution'], 'authorization': 'User explicitly approved the listed settings, reuse of existing safe responses, upload of model weights, and submission to UVA; then approved raising only the task-response limit to 2048 tokens', 'approved_unix': time.time(), 'config_sha256': digest(config_path), 'input_hashes': {str(path.relative_to(ROOT)): digest(path) for path in required}}
    write_json(destination / 'approval.json', approval)
    approved_inputs(config_path, config)
    return {**report, 'approval_recorded': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/curation/configs/config_reference_groups.json')
    parser.add_argument('--record-approval', action='store_true', help='Record the explicit user approval after all targets pass the approved limits')
    args = parser.parse_args()
    print(json.dumps(prepare(args.config.resolve(), args.record_approval), indent=2))
