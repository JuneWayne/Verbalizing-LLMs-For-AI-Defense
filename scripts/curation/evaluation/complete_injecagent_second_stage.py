"""Complete only the 24 approved missing InjecAgent second-stage responses"""

from collections import Counter
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import re

import torch

from scripts.curation.common import ROOT, digest, write_json
from scripts.curation.evaluation.evaluate_task_benchmarks import injecagent_messages
from scripts.curation.evaluation.rescore_injecagent import next_request_score, read_arguments, score_case, summarize
from scripts.curation.evaluation.task_benchmark_common import (
    CONFIG,
    LocalModel,
    METHODS,
    OUTPUT,
    UPSTREAM,
    block_network,
    import_file,
)


REVIEW = ROOT / 'model/artifacts/reports/injecagent_rescoring_20261020_v2/cases.jsonl'
SOURCE = OUTPUT / 'InjecAgent/evaluation'
DESTINATION = OUTPUT / 'InjecAgent/second_stage_recovery_v1'


def prepare():
    rows = [json.loads(line) for line in REVIEW.read_text().splitlines()]
    selected = [row for row in rows if row['reason'] == 'missing_second_stage']
    if len(selected) != 24:
        raise ValueError('The approved run contains exactly 24 missing second stages')
    for entry in selected:
        path = SOURCE / entry['method'] / entry['file']
        row = json.loads(path.read_text())
        if digest(path) != entry['source_sha256'] or len(row['turns']) != 1 or 'stage2' in row:
            raise ValueError(f'Saved response changed or already has a second stage: {path}')
        if score_case(row)['reason'] != 'missing_second_stage':
            raise ValueError(f'The case is not an approved missing second stage: {path}')
    files = [Path(__file__), Path(__file__).with_name('rescore_injecagent.py'), Path(__file__).with_name('evaluate_task_benchmarks.py'), Path(__file__).with_name('task_benchmark_common.py'), CONFIG]
    manifest = {'cases': selected, 'count': 24, 'review_sha256': digest(REVIEW), 'input_hashes': {str(path.relative_to(ROOT)): digest(path) for path in files}, 'generation': {'do_sample': False, 'max_new_tokens': 1024, 'dtype': 'bfloat16', 'seed': 42, 'batch_size': 1}, 'training': False, 'first_stage_regeneration': False, 'simulator': 'Published or existing frozen-base responses, with the existing frozen-base simulator as fallback'}
    path = DESTINATION / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('The prepared recovery run changed')
    write_json(path, manifest)
    print(json.dumps({'prepared': 24, 'models': dict(Counter(row['method'] for row in selected))}), flush=True)


def cached_observation(tool, parameters, published):
    key = f'({tool},{parameters})'
    if key in published:
        return published[key]
    # whitespace or key order can differ while the tool arguments remain identical
    arguments, _, _ = read_arguments(parameters)
    prefix = f'({tool},'
    for saved_key, observation in published.items():
        if saved_key.startswith(prefix) and saved_key.endswith(')'):
            try:
                saved_arguments, _, _ = read_arguments(saved_key[len(prefix):-1])
            except (ValueError, SyntaxError, TypeError):
                continue
            if saved_arguments == arguments:
                return observation
    return None


def complete_case(original, observation, generator, tools, prompts):
    first = score_case(original)
    if first['reason'] != 'missing_second_stage':
        raise ValueError('Only a missing second stage can be completed')
    item = dict(original['item'])
    parameters = json.dumps(first['stage1']['arguments'], ensure_ascii=False)
    item.update({'Attacker Tool Parameters': parameters, 'Attacker Tool Thought': first['stage1']['thought'], 'Attacker Tool Response': observation})
    response = generator.generate(injecagent_messages(item, tools, prompts, stage2=True))
    # preserve the first response exactly and store the continuation in a separate record
    result = {**original, 'original_stage1': original['stage1'], 'stage1': {'eval': 'succ', 'Attacker Tool Parameters': parameters, 'Attacker Tool Thought': first['stage1']['thought']}, 'stage2': {}, 'turns': [original['turns'][0], response], 'simulated_observation': observation}
    score = score_case(result)
    result['stage2'] = {**score['stage2'], 'eval': 'invalid' if score['success'] is None else 'succ' if score['success'] else 'unsucc'}
    result['rescored'] = next_request_score(score)
    result['native_success'] = result['rescored']['success']
    result['repetition'] = any(turn.get('repetitive_at_limit', False) for turn in result['turns'])
    result['unfinished'] = any(not turn['finished'] for turn in result['turns'])
    return result


def run():
    manifest = json.loads((DESTINATION / 'manifest.json').read_text())
    if digest(REVIEW) != manifest['review_sha256']:
        raise ValueError('The source review changed')
    for name, checksum in manifest['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'The prepared run changed: {name}')
    prompts = import_file('injecagent_recovery_prompts', UPSTREAM / 'InjecAgent/src/prompts/agent_prompts.py')
    simulator = import_file('injecagent_recovery_simulator', UPSTREAM / 'InjecAgent/src/prompts/generation_prompts.py')
    published = json.loads((UPSTREAM / 'InjecAgent/data/attacker_simulated_responses.json').read_text())
    tools = {}
    for toolkit in json.loads((UPSTREAM / 'InjecAgent/data/tools.json').read_text()):
        for tool in toolkit['tools']:
            tool = {**tool, 'name': toolkit['toolkit'] + tool['name']}
            tools[tool['name']] = tool
    status = {'status': 'running', 'total': 24, 'completed': 0, 'errors': 0}
    write_json(DESTINATION / 'run.json', status)
    for method in METHODS:
        selected = [row for row in manifest['cases'] if row['method'] == method]
        pending = [row for row in selected if not (DESTINATION / 'evaluation' / method / row['file']).exists()]
        if not pending:
            status['completed'] += len(selected)
            continue
        generator = LocalModel(method)
        expected = json.loads((SOURCE / method / 'run.json').read_text())['model']
        if generator.metadata != expected:
            raise ValueError(f'The model or selected checkpoint changed: {method}')
        block_network()
        status['completed'] += len(selected) - len(pending)
        for entry in pending:
            path = SOURCE / method / entry['file']
            if digest(path) != entry['source_sha256']:
                raise ValueError(f'The first response changed: {path}')
            original = json.loads(path.read_text())
            parameters = json.dumps(entry['stage1']['arguments'], ensure_ascii=False)
            tool = original['item']['Attacker Tools'][0]
            observation = cached_observation(tool, parameters, published)
            source = 'published_cache'
            try:
                if observation is None:
                    key = hashlib.sha256(f'({tool},{parameters})'.encode()).hexdigest() + '.json'
                    existing = OUTPUT / 'InjecAgent/local_simulator' / key
                    cache = DESTINATION / 'local_simulator' / key
                    source = 'frozen_base_qwen_local'
                    if existing.exists() or cache.exists():
                        simulated = json.loads((existing if existing.exists() else cache).read_text())
                    else:
                        prompt = simulator.DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE.format(example=simulator.EXAMPLE, attacker_tool=json.dumps(tools[tool], indent=True), tool_parameters=parameters)
                        simulated = generator.generate([{'role': 'system', 'content': simulator.SYSTEM_MESSAGE}, {'role': 'user', 'content': prompt}], base=True)
                        write_json(cache, simulated)
                    blocks = re.findall(r'```(.*?)```', simulated['text'], re.DOTALL)
                    observation = blocks[0].strip().removeprefix('json').strip() if blocks else ''
                    if not observation or not simulated['finished']:
                        raise ValueError('Simulator did not return a complete fenced response')
                generator.turns = []
                result = complete_case(original, observation, generator, tools, prompts)
                result.update(source_sha256=entry['source_sha256'], simulator_source=source, recovery_model=generator.metadata)
                write_json(DESTINATION / 'evaluation' / method / entry['file'], result)
                status['completed'] += 1
                print(f"{method} {entry['id']}: success={result['native_success']} ({status['completed']}/24)", flush=True)
            except Exception as error:
                status['errors'] += 1
                write_json(DESTINATION / 'errors' / method / entry['file'], {'error': str(error), 'source_sha256': entry['source_sha256']})
                print(f"{method} {entry['id']}: {error}", flush=True)
            write_json(DESTINATION / 'run.json', status)
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    # refresh all scores with only the approved second-stage additions
    rows = [next_request_score(json.loads(line)) for line in REVIEW.read_text().splitlines()]
    for row in rows:
        recovered = DESTINATION / 'evaluation' / row['method'] / row['file']
        if recovered.exists():
            record = json.loads(recovered.read_text())
            row.update(record['rescored'], repetition=record['repetition'], unfinished=record['unfinished'], recovery_file=str(recovered.relative_to(ROOT)))
    report = DESTINATION / 'report'
    report.mkdir(exist_ok=True)
    (report / 'cases.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    summaries = summarize(rows)
    with (report / 'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    status.update(status='complete' if status['completed'] == 24 else 'incomplete', remaining_unscored=sum(row['success'] is None for row in rows))
    write_json(DESTINATION / 'run.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
