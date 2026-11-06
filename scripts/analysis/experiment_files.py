"""Read model weights and save experiment files without loading an experiment runner."""

from pathlib import Path
import hashlib
import json

from safetensors import safe_open


def saved_path(path):
    """Locate a recorded file after relocation without rewriting historical manifests."""
    root = Path(__file__).resolve().parents[2]
    path = Path(path)
    if path.is_absolute():
        try:
            relative = path.relative_to(root)
        except ValueError:
            return path
    else:
        relative = path
    # Original records keep their old paths and hashes as evidence of the completed run.
    if relative.parts and relative.parts[0] == 'models':
        return root / 'model' / Path(*relative.parts[1:])
    if relative.parts and relative.parts[0] == 'results':
        return root / 'model/artifacts' / Path(*relative.parts[1:])
    return root / relative


def file_hash(path):
    """Identify exact file contents so different runs cannot be mixed silently."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read_tensor(model_path, name):
    """Read one weight matrix, including models split across several files."""
    index_path = model_path / 'model.safetensors.index.json'
    if index_path.exists():
        # The index maps each matrix name to the file that stores it.
        weight_index = json.loads(index_path.read_text())
        weight_file = weight_index['weight_map'][name]
    else:
        weight_file = 'model.safetensors'
    with safe_open(model_path / weight_file, framework='pt', device='cpu') as file:
        return file.get_tensor(name)


def save_json(path, value, fixed=False):
    """Save a complete JSON file; reject changes to settings already used in a run."""
    if fixed and path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Existing settings differ: {path}')
    # Replace only after the full write, so an interrupted write cannot damage the old file.
    temporary_path = path.with_suffix('.partial')
    temporary_path.write_text(json.dumps(value, indent=2))
    temporary_path.replace(path)
