"""Replay the released AgentLAB Task-Injection cases without paid attack generation"""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from scripts.curation.evaluation.evaluate_task_benchmarks import dojo_pipeline, run_dojo
from scripts.curation.evaluation.task_benchmark_common import (
    CONFIG,
    LocalModel,
    METHODS,
    OUTPUT,
    ROOT,
    TOOL_PARSER_VERSION,
    UPSTREAM,
    block_network,
    digest,
    summarize,
    write_json,
)


SOURCE = UPSTREAM / 'AgentLAB/Task-Injection/agentdojo'
DESTINATION = OUTPUT / 'AgentLAB'
os.environ['HF_HOME'] = str(ROOT / 'model/benchmark_hf_cache')
sys.path.insert(0, str(SOURCE / 'src'))


def suites():
    # load the suite implementation only after selecting the benchmark source checkout
    get_suite = import_module('agentdojo.task_suite.load_suites').get_suite
    return {name: get_suite('v1.2.1', name) for name in ['workspace', 'travel', 'banking', 'slack']}


def prepare():
    BaseAttack = import_module('agentdojo.attacks.base_attacks').BaseAttack
    attacks = import_module('agentdojo.attacks').important_instructions_attacks

    commit = subprocess.check_output(['git', '-C', str(UPSTREAM / 'AgentLAB'), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '36f58e60c36bbd6d5b8e61d50d7db7d9ea7258d7':
        raise ValueError('Unexpected AgentLAB source version')
    loaded = suites()
    rows, excluded, used_files = [], [], []
    seen_clean = set()
    # use the first saved candidate, never choose an injection by its reported success
    # the original injector reads newest_injection.json, so stage these fixed candidates there
    replay_root = DESTINATION / 'input'
    attacks._find_repo_root = lambda: replay_root
    for name, suite in loaded.items():
        attacker = object.__new__(attacks.Human_ReasoningAttack)
        BaseAttack.__init__(attacker, suite, SimpleNamespace(name='replay'))
        source = SOURCE / 'res/long_horizon' / name / 'v1.2.1'
        for file in sorted(source.glob('*/injection_iter_0_v0.json')):
            match = re.fullmatch(r'(user_task_\d+)_(injection_task_\d+)', file.parent.name)
            if not match:
                excluded.append({'file': str(file.relative_to(ROOT)), 'reason': 'unrecognized_task_folder'})
                continue
            task_id, injection_id = match.groups()
            try:
                task = suite.user_tasks[task_id]
                injection = suite.injection_tasks[injection_id]
                staged = replay_root / 'res/long_horizon/replay' / name / 'v1.2.1' / file.parent.name / 'newest_injection.json'
                staged.parent.mkdir(parents=True, exist_ok=True)
                write_json(staged, json.loads(file.read_text()))
                values = attacker.attack(task, injection)
                # parse the resulting environment now, before allocating a GPU
                suite.load_and_inject_default_environment(values)
            except Exception as error:
                excluded.append({'file': str(file.relative_to(ROOT)), 'reason': f'{type(error).__name__}: {error}'})
                continue
            used_files.append(file)
            if (name, task_id) not in seen_clean:
                rows.append({'id': f'{name}/{task_id}/safe', 'suite': name, 'task': task_id, 'injection_task': None, 'condition': 'safe', 'injections': {}})
                seen_clean.add((name, task_id))
            rows.append({'id': f'{name}/{task_id}/{injection_id}', 'suite': name, 'task': task_id, 'injection_task': injection_id, 'condition': 'injection', 'injections': values, 'payload_source': str(file.relative_to(ROOT)), 'payload_sha256': digest(file)})
    files = [file for file in (SOURCE / 'src').rglob('*') if file.is_file() and '__pycache__' not in file.parts]
    files += [Path(__file__), ROOT / 'scripts/curation/evaluation/task_benchmark_common.py', ROOT / 'scripts/curation/evaluation/evaluate_task_benchmarks.py', ROOT / 'bash/evaluate_agentlab_reference.sbatch', CONFIG]
    manifest = {'benchmark': 'AgentLAB Task-Injection fixed replay', 'source_commit': commit, 'methods': METHODS, 'cases': len(rows), 'cases_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), 'excluded_cases': excluded, 'payload_hashes': {str(file.relative_to(ROOT)): digest(file) for file in used_files}, 'changes': ['Task-Injection track only; Objective-Drifting and Memory-Poisoning are not evaluated by this runner', 'First released candidate for each available task pair; no adaptive generation, rewriting, or success-based selection', 'Original injector, Qwen3-8B tokenizer formatting, environments, and task checks', 'Qwen2.5-7B victim with native tool formatting, greedy 1024 tokens per turn, 15 tool-loop iterations, 32768 context limit', 'These results measure fixed-attack transfer, not the full adaptive AgentLAB benchmark'], 'input_hashes': {str(file.relative_to(ROOT)): digest(file) for file in sorted(files)}}
    manifest['tool_parser_version'] = TOOL_PARSER_VERSION
    manifest['changes'].append('Complete standalone tagged calls accepted after explanatory text; quoted code and incomplete responses do not dispatch tools')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if any(DESTINATION.rglob('case_*.json')):
        raise ValueError('AgentLAB already has results')
    write_json(DESTINATION / 'cases.json', rows)
    write_json(DESTINATION / 'manifest.json', manifest)
    print(f'AgentLAB: {len(rows)} cases prepared, {len(excluded)} excluded with recorded reasons')


def run(method, smoke):
    manifest = json.loads((DESTINATION / 'manifest.json').read_text())
    for name, checksum in manifest['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Frozen AgentLAB input changed: {name}')
    cases = json.loads((DESTINATION / 'cases.json').read_text())
    if hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() != manifest['cases_sha256']:
        raise ValueError('Frozen AgentLAB cases changed')
    if smoke:
        groups = {}
        for case in cases:
            groups.setdefault((case['suite'], case['condition']), case)
        cases = list(groups.values())
    folder = DESTINATION / ('smoke' if smoke else 'evaluation') / method
    folder.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(file.read_text()) for file in sorted(folder.glob('case_*.json'))]
    done = {row['id'] for row in previous}
    generator = LocalModel(method)
    loaded = suites()
    pipeline = dojo_pipeline('AgentLAB', generator)
    block_network()
    status = {'status': 'running', 'model': generator.metadata, 'manifest_sha256': digest(DESTINATION / 'manifest.json'), 'total': len(cases), 'completed': len(previous)}
    if (folder / 'run.json').exists():
        old = json.loads((folder / 'run.json').read_text())
        if old['model'] != status['model'] or old['manifest_sha256'] != status['manifest_sha256']:
            raise ValueError('Cannot combine different AgentLAB runs')
    write_json(folder / 'run.json', status)
    for index, case in enumerate(cases):
        if case['id'] in done:
            continue
        generator.turns = []
        row = {**case, 'native_success': None, 'task_success': None}
        try:
            row.update(run_dojo(case, loaded, pipeline))
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            if 'out of memory' in str(error).lower():
                raise
        row.update(turns=generator.turns, repetition=any(turn['repetitive_at_limit'] for turn in generator.turns), unfinished=any(not turn['finished'] for turn in generator.turns))
        temporary = folder / f'case_{index:05d}.tmp'
        write_json(temporary, row)
        temporary.replace(folder / f'case_{index:05d}.json')
        previous.append(row)
        status['completed'] = len(previous)
        write_json(folder / 'run.json', status)
        write_json(folder / 'metrics.json', summarize(previous))
        print(f'AgentLAB {method}: {len(previous)}/{len(cases)} {case["id"]} error={row.get("error")}', flush=True)
    status['status'] = 'complete'
    write_json(folder / 'run.json', status)
    if smoke and any('error' in row for row in previous):
        raise RuntimeError('AgentLAB smoke tests contain errors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--method-index', type=int, default=0)
    args = parser.parse_args()
    prepare() if args.prepare else run(METHODS[args.method_index], args.smoke)
