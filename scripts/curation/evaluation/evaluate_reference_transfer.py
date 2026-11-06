"""Extend LLMail evaluation and correct the simulated business tool-call parser"""

from pathlib import Path
import argparse
import json
import shutil

from scripts.curation.common import ROOT, digest, read_jsonl, write_json
from scripts.curation.evaluation import (
    evaluate_llmail_reference as llmail_evaluator,
    evaluate_synthetic_business as business_evaluator,
)


REMAINING = ['linear_svm', 'neural_network', 'jlens_llmail', 'jlens_synthetic']
ALL_METHODS = ['base', 'random', 'text_diversity', 'model_failures', 'logistic_regression', *REMAINING]


def parse_business_calls(text):
    # a leading tool marker identifies a call, while quoted examples in prose stay text
    text = text.strip()
    calls = []
    decoder = json.JSONDecoder()
    while text:
        marked = text.startswith('<tool_call>')
        if marked:
            text = text[len('<tool_call>'):].lstrip()
        elif calls or not text.startswith('{'):
            break
        try:
            value, end = decoder.raw_decode(text)
        except ValueError:
            break
        if not isinstance(value, dict) or value.get('name') not in ['submit_request', 'send_message'] or not isinstance(value.get('arguments'), dict):
            break
        trailing = text[end:].lstrip()
        if not marked and trailing:
            break
        # Qwen sometimes omits the closing marker despite generating complete JSON
        # the existing runner still rejects every call from a token-limited response
        calls.append(value)
        if trailing.startswith('</tool_call>'):
            trailing = trailing[len('</tool_call>'):].lstrip()
        text = trailing
    return calls


def evaluation_module(dataset):
    if dataset == 'llmail':
        module = llmail_evaluator
    else:
        module = business_evaluator
    return module


def output_folder(dataset):
    name = 'fipi_qwen25_7b_reference_llmail_remaining_v1' if dataset == 'llmail' else 'fipi_qwen25_7b_reference_business_v2'
    return ROOT / 'model/artifacts/curation' / name


def prepare(dataset):
    module = evaluation_module(dataset)
    original = module.OUTPUT
    manifest_file = original / 'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    for name, checksum in manifest['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Original frozen input changed: {name}')
    methods = REMAINING if dataset == 'llmail' else ALL_METHODS
    fixture = 'prompts.jsonl' if dataset == 'llmail' else 'cases.jsonl'
    rows = read_jsonl(original / fixture)
    config = json.loads((ROOT / 'scripts/curation/configs/config_reference_groups.json').read_text())
    extra_files = [Path(__file__), ROOT / 'bash/evaluate_reference_transfer.sbatch', manifest_file]
    # test the new selection groups for exact prompt overlap before extending LLMail
    normalize = lambda value: ' '.join(value.casefold().split())
    fitting_text = set()
    for method in REMAINING:
        file = ROOT / config['selection_source'] / method / 'pairs.jsonl'
        extra_files.append(file)
        for pair in read_jsonl(file):
            fitting_text.update(normalize(pair[key]) for key in ['clean_data', 'injected_data'])
    if any(normalize(row['source_text']) in fitting_text for row in rows):
        raise ValueError('Testing data overlaps an additional training selection')
    output = output_folder(dataset)
    output.mkdir(exist_ok=False)
    shutil.copyfile(original / fixture, output / fixture)
    extra_files.append(output / fixture)
    manifest.update(methods=methods, parent_manifest=str(manifest_file.relative_to(ROOT)), parent_manifest_sha256=digest(manifest_file))
    manifest['input_hashes'].update({str(file.relative_to(ROOT)): digest(file) for file in extra_files})
    if dataset == 'business':
        manifest['parser_change'] = 'Accept complete JSON following a leading tool_call marker when its closing marker is missing or prose follows it. Never search prose for calls. Token-limited calls still never execute.'
        manifest['comparison'] = 'Rerun base and all eight adapters with the corrected parser; do not combine these results with business_v1'
    else:
        manifest['comparison'] = 'Same frozen prompts, generation, and checker as llmail_v1; only four newly completed adapters are added'
    write_json(output / 'manifest.json', manifest)
    print(f'Prepared {dataset}: {len(rows)} prompts per model, {len(methods)} models')


def run(dataset, index):
    module = evaluation_module(dataset)
    output = output_folder(dataset)
    manifest = json.loads((output / 'manifest.json').read_text())
    # reuse the frozen generation code without editing the earlier experiment
    module.OUTPUT = output
    if dataset == 'business':
        module.parse_calls = parse_business_calls
    module.run(manifest['methods'][index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['llmail', 'business'], required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--method-index', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.dataset)
    else:
        run(args.dataset, args.method_index)
