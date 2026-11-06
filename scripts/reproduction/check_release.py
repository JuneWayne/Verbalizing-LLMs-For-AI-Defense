"""Check a public notebook ZIP without importing model code or starting computation"""

from pathlib import PurePosixPath
import argparse
import ast
import hashlib
import json
import re
import zipfile

from scripts.reproduction.build_release import PUBLIC_LENSES


def check(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('The ZIP contains duplicate paths')
        manifest = json.loads(archive.read('release_manifest.json'))
        if set(names) != set(manifest['files']) | {'release_manifest.json'}:
            raise ValueError('The file list differs from the manifest')
        # A weights release must contain both original checkpoints at their configured paths
        # All other model files and every source dataset remain excluded
        lenses = {name for name in names if name.startswith('model/')}
        if lenses and lenses != set(PUBLIC_LENSES):
            raise ValueError('The weights package must contain exactly the two public Jlens checkpoints')
        if set(manifest.get('included_lenses', [])) != lenses:
            raise ValueError('Lens files differ from the release manifest')
        notebooks = 0
        for name, checksum in manifest['files'].items():
            parts = PurePosixPath(name).parts
            if name.startswith('/') or '..' in parts or parts[0] in ['data', '.git', 'bash'] or (parts[0] == 'model' and name not in PUBLIC_LENSES) or (parts[0] == '.github' and name != '.github/workflows/check-research.yml'):
                raise ValueError(f'Unexpected release path: {name}')
            if name in PUBLIC_LENSES:
                # Read large tensor files in chunks while checking that their bytes are unchanged
                with archive.open(name) as stream:
                    checksum_actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if checksum_actual != checksum:
                    raise ValueError(f'Lens contents changed: {name}')
                continue
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f'File contents changed: {name}')
            if name.endswith('.py'):
                ast.parse(data.decode())
            if not name.endswith('.ipynb'):
                continue
            notebooks += 1
            notebook = json.loads(data)
            if notebook['nbformat'] != 4 or not notebook['cells']:
                raise ValueError(f'Invalid notebook: {name}')
            if not ''.join(notebook['cells'][0]['source']).startswith('# '):
                raise ValueError(f'Missing notebook introduction: {name}')
            headings = set()
            for cell in notebook['cells']:
                source = ''.join(cell['source'])
                if cell.get('attachments'):
                    raise ValueError(f'Notebook contains hidden attachments: {name}')
                if any(output.get('output_type') == 'error' for output in cell.get('outputs', [])):
                    raise ValueError(f'Notebook contains an execution error: {name}')
                if cell['cell_type'] == 'code':
                    ast.parse(source)
                    # Every code section must have run, including sections that only import packages
                    if not isinstance(cell.get('execution_count'), int) or cell['execution_count'] < 1:
                        raise ValueError(f'Notebook contains an unexecuted code section: {name}')
                if source.startswith('## '):
                    headings.add(source.splitlines()[0][3:].replace(' ', '-'))
            links = set(re.findall(r'\]\(#([^)]*)\)', ''.join(notebook['cells'][0]['source'])))
            if links != headings:
                raise ValueError(f'Contents links do not match headings: {name}')
            audit = json.loads(archive.read('scripts/reproduction/notebook_audit.json'))
            record = next(row for row in audit['notebooks'] if row['notebook'] == name)
            source = [{'type': cell['cell_type'], 'source': ''.join(cell['source'])} for cell in notebook['cells']]
            checksum = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
            if record['status'] != 'passed' or checksum != record['source_sha256']:
                raise ValueError(f'Notebook execution record is missing or outdated: {name}')
            # Match the published tables and figures to the outputs from the successful run
            # This catches cleared or edited outputs even when the source code stays the same
            outputs = [{'execution_count': cell.get('execution_count'), 'outputs': cell.get('outputs', [])} for cell in notebook['cells'] if cell['cell_type'] == 'code']
            checksum = hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest()
            if checksum != record.get('execution_sha256') or not any(cell['outputs'] for cell in outputs):
                raise ValueError(f'Notebook outputs are missing or differ from its execution record: {name}')
        return {'notebooks_checked': notebooks, 'files_checked': len(manifest['files']), 'lenses_checked': len(lenses), 'model_computation': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip')
    print(json.dumps(check(parser.parse_args().zip), indent=2))
