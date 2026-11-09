"""Check a reference-response run and record its inputs without private approval files"""

from pathlib import Path
import json
import time

from scripts.curation.common import METHODS, ROOT, digest, write_json


TRAINING_CODE_FILES = [
    'scripts/curation/training/reference_loss.py',
    'scripts/curation/training/training_helpers.py',
    'scripts/curation/training/run_manifest.py',
    'scripts/curation/training/train_reference_groups.py',
    'scripts/curation/common.py',
    'scripts/curation/evaluation/evaluate.py',
    'scripts/curation/evaluation/evaluate_fipi.py',
    'scripts/curation/preparation/Record_hidden_state_vectors_and_model_response.py',
]


# Read settings from the configuration passed to the Python command
# The checks below cover the options implemented by this trainer
# Execution location, scheduler files and old conversation approvals have no role here
def read_config(path):
    config = json.loads(Path(path).read_text())
    methods = config['methods']
    if not methods or len(methods) != len(set(methods)) or any(method not in METHODS for method in methods):
        raise ValueError('Choose distinct selection methods from the supported method names')

    # The recorded experiment uses one pair and one optimizer update per training step
    # Other batching settings need an implementation change before they can be used
    if config['pairs_per_update'] != 1 or config['gradient_accumulation'] != 1:
        raise ValueError('This trainer supports one pair per update and gradient accumulation of one')
    if config['optimizer'] != 'AdamW' or config['compute_dtype'] != 'bfloat16' or config.get('quantization') is not None:
        raise ValueError('This trainer uses AdamW and BF16 without quantization')
    if config.get('separation_loss_enabled'):
        raise ValueError('This trainer uses reference-response loss only')

    for field in ['training_pairs', 'validation_pairs', 'testing_pairs', 'epochs', 'rank', 'max_input_tokens', 'max_task_response_tokens', 'evaluation_max_new_tokens']:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if config['learning_rate'] <= 0 or not 0 <= config['dropout'] < 1:
        raise ValueError('Use a positive learning rate and dropout between zero and one')
    if not 0 < config['warmup_fraction'] < 1 or not 0 <= config['final_learning_rate_fraction'] <= 1:
        raise ValueError('The learning-rate schedule fractions are outside their supported ranges')

    # Check the requested layers against the downloaded model configuration
    # Loading this JSON file does not load model weights or start training
    model = ROOT / config['model']
    architecture = json.loads((model / 'config.json').read_text())
    layers = config['layers']
    if not layers or len(layers) != len(set(layers)) or any(type(layer) is not int or not 0 <= layer < architecture['num_hidden_layers'] - 1 for layer in layers):
        raise ValueError('A configured layer is outside the supported block range')
    return config


# Record which configuration, prepared pairs, source code and model files a run uses
# This manifest is created automatically when training starts
# It identifies the inputs without granting permission or requiring a particular cluster
def record_manifest(config_path, config):
    data = ROOT / config['data']
    files = {Path(config_path).resolve(), data / 'validation.jsonl', ROOT / config['validation_source'], ROOT / config['testing_source']}
    files.update(data / 'training' / f'{method}.jsonl' for method in config['methods'])
    files.update(ROOT / config['selection_source'] / method / 'pairs.jsonl' for method in config['methods'])
    files.update(ROOT / name for name in TRAINING_CODE_FILES)

    # Hash the actual downloaded model files instead of requiring an old model-hash manifest
    model = ROOT / config['model']
    model_files = sorted(path for path in model.iterdir() if path.is_file() and path.suffix in ['.safetensors', '.bin', '.json', '.jinja', '.model'])
    if not any(path.suffix in ['.safetensors', '.bin'] for path in model_files):
        raise FileNotFoundError('Download the model weights before starting a training run')
    input_hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest(path) for path in sorted(files)}
    model_hashes = {path.name: digest(path) for path in model_files}
    record = {'config': config, 'input_hashes': input_hashes, 'model_hashes': model_hashes}

    # All eight adapters can share one manifest because only the selected method changes
    # Changed settings or inputs need a separate output folder to preserve earlier results
    folder = ROOT / config['output']
    path = folder / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(key) != value for key, value in record.items()):
            raise ValueError('This output folder belongs to different inputs or settings; choose a new output folder')
        return previous
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('The output folder already contains results; choose a new output folder')
    record['created_unix'] = time.time()
    write_json(path, record)
    return record
