"""Bind a reference-response training run to approved settings, source code and prepared pairs"""

from pathlib import Path
import argparse
import json
import time

from scripts.curation.common import ROOT, digest, write_json
from scripts.curation.training.data_approval import TRAINING_CODE_FILES
from scripts.curation.training.training_helpers import inspect_data


# Create the run manifest only after exact settings and final data are approved
# The manifest records current code, prepared reference targets, testing pairs and model hashes
# Existing training outputs prevent replacing an earlier run's approval
def prepare(path):
    config = json.loads(path.read_text())

    # settings must already carry explicit approval before a run manifest can be created
    if config['status'] != 'approved':
        raise ValueError('Training settings need explicit approval')
    files = {path, ROOT / config['model_hashes']}

    # include shared helpers because a change there can change either experiment
    files.update(ROOT / name for name in TRAINING_CODE_FILES)
    files.add(Path(__file__))
    if 'methods' not in config:
        raise ValueError('This approval script supports the reference-response experiment')

    # inspect all prepared groups and testing splits before recording their content hashes
    groups, validation, testing = inspect_data(config)
    files.add(ROOT / config['data'] / 'validation.jsonl')
    files.update(ROOT / config['data'] / 'training' / f'{method}.jsonl' for method in config['methods'])
    files.update(ROOT / config['selection_source'] / method / 'pairs.jsonl' for method in config['methods'])
    files.update([ROOT / config['validation_source'], ROOT / config['testing_source'], ROOT / 'bash/train_reference_groups.sbatch'])
    print(f'Checked {len(groups)} groups, {len(validation)} validation pairs, {len(testing)} testing pairs')
    folder = ROOT / config['output']
    if folder.exists() and any(file.name not in ('manifest.json', 'approval.json') for file in folder.iterdir()):
        raise FileExistsError('The run already has outputs; approval cannot be replaced')
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'config': config, 'input_hashes': {str(file.relative_to(ROOT)): digest(file) for file in sorted(files)}, 'created_unix': time.time()}
    write_json(folder / 'manifest.json', manifest)
    write_json(folder / 'approval.json', {'training_authorized': True, 'execution': 'UVA HPC', 'manifest_sha256': digest(folder / 'manifest.json'), 'approval': 'User: ok approved, following the exact settings proposal and validation-checkpoint selection explanation', 'reference_targets': config.get('reference_review'), 'created_unix': time.time()})


# Require the same approved configuration, manifest, code and model weights before training
# Any changed file makes the existing approval invalid and requires a newly reviewed run
def verify(path):
    config = json.loads(path.read_text())
    folder = ROOT / config['output']
    approval = json.loads((folder / 'approval.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    if config['status'] != 'approved' or config != manifest['config'] or approval.get('training_authorized') is not True or approval.get('execution') != 'UVA HPC' or approval['manifest_sha256'] != digest(folder / 'manifest.json'):
        raise ValueError('The settings or approval changed')

    # compare each approved input with its current contents, including the loss and training code
    for name, checksum in manifest['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Approved input changed: {name}')
    for name, checksum in json.loads((ROOT / config['model_hashes']).read_text()).items():
        if digest(ROOT / config['model'] / name) != checksum:
            raise ValueError(f'Original model file changed: {name}')
    return config


# This command records a reviewed manifest and does not train an adapter
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', required=True)
    args = parser.parse_args()
    prepare(args.config.resolve())
