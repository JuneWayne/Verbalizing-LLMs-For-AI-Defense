"""Evaluate the BIPIA subset with published rules and a separate local judge"""

from importlib import import_module
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import types

from accelerate import PartialState

from scripts.curation.evaluation.task_benchmark_common import (
    CONFIG,
    LocalModel,
    METHODS,
    OUTPUT,
    ROOT,
    UPSTREAM,
    block_network,
    digest,
    write_json,
)


if os.environ.get('BIPIA_DEPS'):
    sys.path.append(os.environ['BIPIA_DEPS'])
SOURCE = UPSTREAM / 'BIPIA'
DESTINATION = OUTPUT / 'BIPIA'
sys.path.insert(0, str(SOURCE))


def prepare():
    AutoPIABuilder = import_module('bipia.data').AutoPIABuilder
    insert_end = import_module('bipia.data.utils').insert_end
    insert_start = import_module('bipia.data.utils').insert_start
    insert_middle = import_module('bipia.data.utils').insert_middle

    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != 'a004b69ec0dd446e0afd461d98cb5e96e120a5d0':
        raise ValueError('Unexpected BIPIA source version')
    rows = []
    for task in ['email', 'table', 'code']:
        builder = AutoPIABuilder.from_name(task)(seed=2023)
        context_file = SOURCE / 'benchmark' / task / 'test.jsonl'
        contexts = [json.loads(line) for line in context_file.read_text().splitlines()]
        attack_file = SOURCE / 'benchmark' / ('code_attack_test.json' if task == 'code' else 'text_attack_test.json')
        attacks = json.loads(attack_file.read_text())
        # keep all clean contexts, then cover every injection at each published position
        # this is a fixed subset, not the full context-by-injection cross product
        order = list(range(len(contexts)))
        random.Random(2023).shuffle(order)
        for index, context in enumerate(contexts):
            example = dict(context)
            if task == 'code':
                example = {key: '\n'.join(value) if isinstance(value, list) else value for key, value in example.items()}
            system, user = builder.construct_prompt(example, require_system_prompt=True)
            rows.append({'id': f'{task}/safe/{index}', 'task': task, 'condition': 'safe', 'context_index': index, 'ideal': example['ideal'], 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
        count = 0
        for category, payloads in attacks.items():
            for variant, payload in enumerate(payloads):
                for position, insert in [('end', insert_end), ('start', insert_start), ('middle', insert_middle)]:
                    index = order[count % len(order)]
                    count += 1
                    example = dict(contexts[index])
                    if task == 'code':
                        example = {key: '\n'.join(value) if isinstance(value, list) else value for key, value in example.items()}
                    example['context'] = insert(example['context'], payload, random_state=2023)
                    system, user = builder.construct_prompt(example, require_system_prompt=True)
                    rows.append({'id': f'{task}/{category}-{variant}/{position}', 'task': task, 'condition': 'injection', 'context_index': index, 'category': category, 'attack_name': f'{category}-{variant}', 'position': position, 'ideal': example['ideal'], 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
    files = [file for file in SOURCE.rglob('*') if file.is_file() and '.git' not in file.parts and '__pycache__' not in file.parts]
    files += [Path(__file__), ROOT / 'scripts/curation/evaluation/task_benchmark_common.py', CONFIG]
    manifest = {'benchmark': 'BIPIA subset with local judge', 'source_commit': commit, 'methods': METHODS, 'cases': len(rows), 'cases_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), 'generation': {'max_new_tokens': 1024, 'max_context_tokens': 32768, 'do_sample': False, 'seed': 42, 'dtype': 'bfloat16'}, 'changes': ['EmailQA, TableQA and CodeQA only; WebQA and Summarization source contexts are not included in this release', 'Every published testing injection at start, middle and end, with one fixed context each; not the complete cross product', 'Original prompts and rule checks; paid GPT judge replaced with frozen base Qwen2.5-7B', 'Rule-check ASR and local-judge ASR are separate; local judgments require human review', 'No generated code executes and no real tools are called', 'Task completion remains unscored because answer similarity is not a verified task-success checker'], 'input_hashes': {str(file.relative_to(ROOT)): digest(file) for file in sorted(files)}}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if any(DESTINATION.rglob('case_*.json')):
        raise ValueError('This benchmark already has results')
    write_json(DESTINATION / 'cases.json', rows)
    write_json(DESTINATION / 'manifest.json', manifest)
    print(f'BIPIA: {len(rows)} cases prepared')


def checkers(generator):
    # BIPIA dependencies can live in their own directory, configured before this call
    DetectorFactory = import_module('langdetect').DetectorFactory

    PartialState(cpu=True)
    DetectorFactory.seed = 2023

    # replace only the deprecated paid client, keeping the released judge questions
    class LocalJudge:
        def __init__(self, config, judge_fn, format_fn):
            self.format_fn = format_fn
            self.last_response = None

        def _compute_score(self, prediction, **kwargs):
            response = generator.generate(self.format_fn(prediction, chat=True), base=True)
            self.last_response = response
            answer = response['text'].strip().lower().rstrip('.')
            if not response['finished'] or answer not in ['yes', 'no']:
                return -1
            return int(answer == 'yes')

    module = types.ModuleType('bipia.metrics.eval.model')
    module.ModelEval = LocalJudge
    sys.modules[module.__name__] = module
    # register the published checks after installing the local judge replacement
    depia_regist_fn = import_module('bipia.metrics.regist').depia_regist_fn
    return {name: factory() for name, factory in depia_regist_fn({'chat': True}).items()}, LocalJudge


def metrics(rows):
    result = {'cases': len(rows), 'task_completion_rate': None, 'generation_errors': sum('error' in row for row in rows)}
    for kind in ['published_rule', 'local_base_model_judge']:
        group = [row for row in rows if row.get('checker') == kind]
        scored = [row for row in group if row.get('injection_success') is not None]
        result[kind] = {'cases': len(group), 'scored_cases': len(scored), 'injection_success_rate': sum(row['injection_success'] for row in scored) / len(scored) if scored else None, 'repetition': sum(row.get('repetition', False) for row in group), 'success_or_repetition_rate': sum(row['injection_success'] or row.get('repetition', False) for row in scored) / len(scored) if scored else None}
    return result


def run(method, smoke):
    manifest = json.loads((DESTINATION / 'manifest.json').read_text())
    for name, checksum in manifest['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Frozen BIPIA input changed: {name}')
    cases = json.loads((DESTINATION / 'cases.json').read_text())
    if hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() != manifest['cases_sha256']:
        raise ValueError('Frozen BIPIA cases changed')
    if smoke:
        groups = {}
        for case in cases:
            groups.setdefault((case['task'], case['condition']), case)
        cases = list(groups.values())
    folder = DESTINATION / ('smoke' if smoke else 'evaluation') / method
    folder.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(file.read_text()) for file in sorted(folder.glob('case_*.json'))]
    done = {row['id'] for row in previous}
    generator = LocalModel(method)
    checker_map, judge_type = checkers(generator)
    status = {'status': 'running', 'model': generator.metadata, 'manifest_sha256': digest(DESTINATION / 'manifest.json'), 'total': len(cases), 'completed': len(previous)}
    if (folder / 'run.json').exists():
        old = json.loads((folder / 'run.json').read_text())
        if old['model'] != status['model'] or old['manifest_sha256'] != status['manifest_sha256']:
            raise ValueError('Cannot combine different BIPIA runs')
    write_json(folder / 'run.json', status)
    block_network()
    for index, case in enumerate(cases):
        if case['id'] in done:
            continue
        row = {**case, 'injection_success': None, 'task_success': None}
        generator.turns = []
        try:
            response = generator.generate(case['messages'])
            row.update(response=response, repetition=response['repetitive_at_limit'])
            if case['condition'] == 'injection':
                checker = checker_map[case['attack_name']]
                row['checker'] = 'local_base_model_judge' if isinstance(checker, judge_type) else 'published_rule'
                score = checker._compute_score(prediction=response['text'], reference=case['ideal'])
                row['injection_success'] = bool(score) if score in [0, 1] else None
                if isinstance(checker, judge_type):
                    row['judge_response'] = checker.last_response
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            if 'out of memory' in str(error).lower():
                raise
        temporary = folder / f'case_{index:05d}.tmp'
        write_json(temporary, row)
        temporary.replace(folder / f'case_{index:05d}.json')
        previous.append(row)
        status['completed'] = len(previous)
        write_json(folder / 'run.json', status)
        write_json(folder / 'metrics.json', metrics(previous))
        print(f'BIPIA {method}: {len(previous)}/{len(cases)} {case["id"]} error={row.get("error")}', flush=True)
    status['status'] = 'complete'
    write_json(folder / 'run.json', status)
    if smoke and any('error' in row for row in previous):
        raise RuntimeError('BIPIA smoke tests contain errors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--method-index', type=int, default=0)
    args = parser.parse_args()
    prepare() if args.prepare else run(METHODS[args.method_index], args.smoke)
