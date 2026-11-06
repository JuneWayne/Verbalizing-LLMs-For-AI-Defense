"""Build a GitHub upload ZIP with readable notebooks and the code they use

Public copies retain executed outputs, including tables, examples and plots
Hidden attachments and widget metadata are removed from the copies
README contains the notebook order, reproduction instructions and script explanations
An optional weights package includes the two trained Jlens checkpoints
Datasets, base model weights, adapters, training approvals and personal execution files stay outside the ZIP
"""

from pathlib import Path
import argparse
import ast
import hashlib
import json
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_LENSES = (
    'model/artifacts/qwen2.5-7b/llmail-v1/lens.pt',
    'model/artifacts/qwen2.5-7b/old-classifier-titlecase-v1/lens.pt',
)

# These are research entry points rather than every script from earlier troubleshooting
ENTRY_POINTS = [
    'scripts/Jacobian_Lens_Fitting.py', 'scripts/curate_jlens_training.py',
    'scripts/data/inspect_fipi.py', 'scripts/data/prepare_qwen_family_pairs.py',
    'scripts/data/prepare_llmail_probe_transfer.py', 'scripts/data/prepare_qwen_llmail.py',
    'scripts/curation/preparation/pool.py', 'scripts/curation/preparation/Record_hidden_state_vectors_and_model_response.py', 'scripts/curation/selection/data_curation_selection.py',
    'scripts/curation/preparation/preparation.py', 'scripts/curation/selection/extract_results.py',
    'scripts/curation/preparation/finalize_training.py', 'scripts/curation/preparation/export_validation.py',
    'scripts/curation/preparation/prepare_lora_groups.py', 'scripts/curation/training/train_reference_groups.py',
    'scripts/curation/training/approved_training_runs.py',
    'scripts/curation/evaluation/evaluate_llmail_reference.py', 'scripts/curation/evaluation/evaluate_reference_transfer.py',
    'scripts/curation/evaluation/evaluate_synthetic_business.py', 'scripts/curation/evaluation/evaluate_task_benchmarks.py',
    'scripts/curation/evaluation/evaluate_agentlab_reference.py', 'scripts/curation/evaluation/evaluate_bipia_reference.py',
    'scripts/curation/evaluation/repair_task_evaluations.py', 'scripts/curation/evaluation/resolve_business_cases.py',
    'scripts/curation/evaluation/rescore_injecagent.py', 'scripts/curation/evaluation/complete_injecagent_second_stage.py',
    'scripts/curation/evaluation/rescore_bipia.py', 'scripts/curation/evaluation/resolve_bipia.py',
    'scripts/curation/evaluation/review_bipia_unscored.py',
    'scripts/analysis/run_reasoning_detectors.py',
    'scripts/reproduction/build_release.py', 'scripts/reproduction/check_release.py',
]


def script_dependencies(paths):
    """Follow local Python imports without importing or executing the scripts"""
    found = set()
    pending = list(paths)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(name)
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module, *[node.module + '.' + alias.name for alias in node.names]]
            for module in modules:
                if not module.startswith('scripts.'):
                    continue
                relative = module.replace('.', '/')
                for candidate in [relative + '.py', relative + '/__init__.py']:
                    if (ROOT / candidate).is_file():
                        pending.append(candidate)
    return found


def public_notebook(path):
    """Keep executed outputs while removing hidden attachments and widget metadata"""
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        cell.pop('attachments', None)
        # Preserve tables, prompt examples, plots and execution counts in every notebook
        # Keep folding settings without exporting widget state or execution timestamps
        cell['metadata'] = {key: value for key, value in cell.get('metadata', {}).items() if key in ['jp-MarkdownHeadingCollapsed', 'tags']}
    notebook['metadata'] = {key: value for key, value in notebook.get('metadata', {}).items() if key in ['kernelspec', 'language_info']}
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + '\n').encode()


def build(destination, include_lenses=False):
    """Package selected source files and public counts without publishing to GitHub"""
    files = script_dependencies(ENTRY_POINTS)
    files.update(str(path.relative_to(ROOT)) for path in (ROOT / 'notebooks').rglob('*.ipynb') if '.ipynb_checkpoints' not in path.parts)
    files.update([
        'README.md', 'pyproject.toml', 'uv.lock', '.gitignore',
        'notebooks/results/defense_evaluations.json',
        'scripts/data/download_datasets.sh',
        'scripts/reproduction/lens_settings.json',
        'scripts/reproduction/notebook_audit.json',
        '.github/workflows/check-research.yml',
    ])
    # Publish only the current selection and reference-response configurations
    files.update(str(path.relative_to(ROOT)) for path in (ROOT / 'scripts/curation/configs').glob('*.json'))
    files.update(str(path.relative_to(ROOT)) for path in (ROOT / 'scripts/curation/tests').glob('test_*.py'))
    files = script_dependencies([name for name in files if name.endswith('.py')]) | files
    contents = {}
    for name in sorted(files):
        path = ROOT / name
        if path.is_symlink():
            raise ValueError(f'Release files must not be symlinks: {name}')
        data = public_notebook(path) if path.suffix == '.ipynb' else path.read_bytes()
        if path.suffix == '.json' and name.startswith('scripts/curation/'):
            # Historical settings remain examples, not approval to launch a new run
            config = json.loads(data)
            original = dict(config)
            for key in ['account', 'modal_profile', 'profile', 'workspace']:
                config.pop(key, None)
            if config.get('status') == 'approved':
                config['status'] = 'requires_review'
            if config != original:
                data = (json.dumps(config, indent=2) + '\n').encode()
        contents[name] = data

    manifest = {'purpose': 'Research release with executed notebook outputs', 'files': {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}, 'included_lenses': list(PUBLIC_LENSES) if include_lenses else [], 'excluded': ['source datasets', 'base model weights and adapters', 'notebook attachments and widget state', 'training approval records', 'cloud account runners', 'personal HPC jobs']}
    if include_lenses:
        # Include only these two approved lens files, never the rest of the model folder
        # Hash each file in chunks so the large matrices do not fill system memory
        for name in PUBLIC_LENSES:
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Missing or linked lens checkpoint: {name}')
            with path.open('rb') as stream:
                manifest['files'][name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a rerun from replacing a release already reviewed
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        if include_lenses:
            # Stream the checkpoints into the ZIP without recompressing their tensor storage
            for name in PUBLIC_LENSES:
                archive.write(ROOT / name, name, compress_type=zipfile.ZIP_STORED)
        archive.writestr('release_manifest.json', json.dumps(manifest, indent=2) + '\n')
    return {'zip': str(destination), 'notebooks': sum(name.endswith('.ipynb') for name in contents), 'files': len(manifest['files']), 'lenses': len(manifest['included_lenses']), 'bytes': destination.stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-lenses', action='store_true', help='Include both trained Jlens checkpoints for a release download')
    args = parser.parse_args()
    print(json.dumps(build(args.output, include_lenses=args.include_lenses), indent=2))
