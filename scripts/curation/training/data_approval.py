"""Verify that prepared reference answers, model files and settings match an approved data run"""

from pathlib import Path
import json

from transformers import AutoTokenizer

from scripts.curation.common import METHODS, ROOT, digest, read_jsonl
from scripts.curation.training.training_helpers import check_rows, check_sources, check_splits


# Bind prepared-data approval to the current loss, schedule, data checks and model input code
TRAINING_CODE_FILES = ['scripts/curation/training/reference_loss.py', 'scripts/curation/training/training_helpers.py', 'scripts/curation/training/data_approval.py', 'scripts/curation/training/train_reference_groups.py', 'scripts/curation/common.py', 'scripts/curation/evaluation/evaluate.py', 'scripts/curation/evaluation/evaluate_fipi.py', 'scripts/curation/preparation/Record_hidden_state_vectors_and_model_response.py']


# Verify the approved configuration and every recorded source file before accepting training data
# The data approval records the exact selected pairs and accepted reference responses
# Current code hashes are checked together with the original model weight hashes
# This check returns the prepared training groups and validation pairs without fitting a model
def approved_inputs(config_path, config):

    data = ROOT / config['data']
    if config['status'] != 'approved':
        raise ValueError('The new training settings require explicit approval')
    approval = json.loads((data / 'approval.json').read_text())
    if approval.get('training_authorized') is not True or approval.get('targets_accepted') is not True:
        raise ValueError('The new settings and response targets require explicit approval')
    if approval.get('execution') != config['execution'] or approval.get('config_sha256') != digest(config_path):
        raise ValueError('The approved execution location or settings differ')
    required = [data / 'validation.jsonl', ROOT / config['model_hashes'], *[data / 'training' / f'{method}.jsonl' for method in METHODS]]
    required.extend(ROOT / config['selection_source'] / method / 'pairs.jsonl' for method in METHODS)
    required.append(ROOT / config['validation_source'])

    # bind approval to the code as well as the data so later edits cannot change a run silently
    required.extend(ROOT / name for name in TRAINING_CODE_FILES)
    required.append(ROOT / 'bash/train_reference_groups.sbatch')
    for path in required:
        name = str(path.relative_to(ROOT))
        if approval.get('input_hashes', {}).get(name) != digest(path):
            raise ValueError(f'Approved file differs: {name}')

    # verify the original model weights separately from the selected prompt and response records
    model_hashes = json.loads((ROOT / config['model_hashes']).read_text())
    if not model_hashes or not any(name.endswith('.safetensors') for name in model_hashes):
        raise ValueError('The model manifest must include the original model weights')
    for name, checksum in model_hashes.items():
        if digest(ROOT / config['model'] / name) != checksum:
            raise ValueError(f'Original model file differs: {name}')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config['model'], local_files_only=True)
    model_config = json.loads((ROOT / config['model'] / 'config.json').read_text())
    if any(layer < 0 or layer >= model_config['num_hidden_layers'] - 1 for layer in config['layers']):
        raise ValueError('A configured training layer is outside the supported block range')

    # apply the same token and source checks to the shared validation set and every selected group
    validation = read_jsonl(data / 'validation.jsonl')
    check_rows(validation, config['validation_pairs'], config, tokenizer.eos_token_id, model_config['vocab_size'])
    check_sources(validation, read_jsonl(ROOT / config['validation_source']))
    groups = {}
    for method in METHODS:
        groups[method] = read_jsonl(data / 'training' / f'{method}.jsonl')
        check_rows(groups[method], config['training_pairs'], config, tokenizer.eos_token_id, model_config['vocab_size'])
        check_sources(groups[method], read_jsonl(ROOT / config['selection_source'] / method / 'pairs.jsonl'), exact=True)
        check_splits(groups[method], validation)
    return groups, validation
