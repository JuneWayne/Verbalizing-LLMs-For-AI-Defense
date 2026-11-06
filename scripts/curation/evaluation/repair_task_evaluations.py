"""Regenerate interrupted tool conversations while preserving unaffected results"""

from pathlib import Path
import argparse
import copy
import json
import time

import torch

from scripts.curation.common import read_jsonl
from scripts.curation.evaluation.evaluate import repetition_check
from scripts.curation.evaluation.evaluate_agentlab_reference import suites
from scripts.curation.evaluation.evaluate_reference_transfer import parse_business_calls
from scripts.curation.evaluation.evaluate_synthetic_business import summarize as business_summary
from scripts.curation.evaluation.evaluate_task_benchmarks import dojo_pipeline, load_dojo, run_dojo
from scripts.curation.evaluation.synthetic_business_rules import execute_calls, score_case
from scripts.curation.evaluation.task_benchmark_common import (
    LocalModel,
    METHODS,
    OUTPUT,
    ROOT,
    TOOL_PARSER_VERSION,
    block_network,
    digest,
    invalid_json_constant,
    parse_tool_calls,
    summarize,
    unique_json_fields,
    write_json,
)
from scripts.curation.evaluation.tool_trace_checks import check_initial_results, completion_status


DESTINATION = ROOT / 'model/artifacts/curation/tool_parser_repair_v3'
BENCHMARKS = ['AgentDyn', 'AgentDojo', 'AgentLAB', 'SyntheticBusiness']


def business_calls(text, finished=True):
    # preserve the business protocol's whole-message JSON form
    if '<tool_call>' not in text:
        if not text.lstrip().startswith('{'):
            return [], False
        if not finished:
            return [], True
        try:
            call = json.loads(text, object_pairs_hook=unique_json_fields, parse_constant=invalid_json_constant)
        except ValueError:
            return [], True
        calls, invalid = [call], False
    else:
        calls, invalid = parse_tool_calls(text, finished=finished)
    if any(not isinstance(call, dict) or call.get('name') not in ['submit_request', 'send_message'] or not isinstance(call.get('arguments'), dict) for call in calls):
        return [], True
    return calls, invalid


def repair_reason(row, business=False):
    if row is None:
        return 'missing_result'
    if row.get('error') or not row.get('turns'):
        return 'evaluation_error'
    turns = row['turns']
    assistants = [message for message in row.get('trace', []) if message['role'] == 'assistant']
    if not business and len(assistants) != len(turns):
        return 'response_trace_mismatch'
    for index, turn in enumerate(turns):
        text = turn['text']
        finished = turn.get('finished', False)
        calls, invalid = business_calls(text, finished) if business else parse_tool_calls(text, finished=finished)
        if invalid:
            return 'ambiguous_or_incomplete_call'
        if business:
            old = parse_business_calls(text) if finished else []
        else:
            old = [{'name': call['function'], 'arguments': call['args']} for call in assistants[index].get('tool_calls', [])]
        if calls != old:
            return 'changed_tool_calls'
    return None


def source_folder(benchmark):
    return ROOT / 'model/artifacts/curation/fipi_qwen25_7b_reference_business_v2' if benchmark == 'SyntheticBusiness' else OUTPUT / benchmark


def prepare(benchmark):
    source = source_folder(benchmark)
    parent = json.loads((source / 'manifest.json').read_text())
    # the original code remains untouched on HPC, so its data and provenance can still be checked
    for name, checksum in parent['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Original frozen input changed: {name}')
    business = benchmark == 'SyntheticBusiness'
    cases = read_jsonl(source / 'cases.jsonl') if business else json.loads((source / 'cases.json').read_text())
    key = 'example_id' if business else 'id'
    output = DESTINATION / benchmark
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Repair plan already exists: {output}')
    counts = {}
    source_hashes = {}
    for method in METHODS:
        if business:
            file = source / method / 'responses.jsonl'
            rows = read_jsonl(file) if file.exists() else []
            if file.exists():
                source_hashes[str(file)] = digest(file)
        else:
            files = sorted((source / 'evaluation' / method).glob('case_*.json'))
            rows = [json.loads(file.read_text()) for file in files]
            source_hashes.update({str(file): digest(file) for file in files})
        previous = {row[key]: row for row in rows}
        plan = []
        for index, case in enumerate(cases):
            row = previous.get(case[key])
            reason = repair_reason(row, business)
            plan.append({'index': index, 'id': case[key], 'reason': reason})
            if reason is None:
                retained = copy.deepcopy(row)
                retained['repair_provenance'] = {'action': 'retained', 'reason': 'Tool parsing is unchanged', 'parent': str(source)}
                write_json(output / 'evaluation' / method / f'case_{index:05d}.json', retained)
        counts[method] = {'retained': sum(item['reason'] is None for item in plan), 'regenerate': sum(item['reason'] is not None for item in plan)}
        write_json(output / 'plans' / f'{method}.json', plan)
    write_json(output / 'cases.json', cases)
    code = [Path(__file__), Path(__file__).with_name('task_benchmark_common.py'), Path(__file__).with_name('evaluate_task_benchmarks.py')]
    manifest = {
        'benchmark': benchmark, 'parser': TOOL_PARSER_VERSION, 'counts': counts,
        'parent': str(source), 'parent_manifest_sha256': digest(source / 'manifest.json'),
        'cases_sha256': digest(output / 'cases.json'), 'source_result_hashes': source_hashes,
        'code_hashes': {str(file): digest(file) for file in code},
        'original_input_hashes': parent['input_hashes'],
        'generation': 'Unchanged model, adapters, prompts, greedy decoding, BF16 and seed 42',
        'limits': 'SyntheticBusiness: 1024 tokens total, 3 turns. Agent benchmarks: 1024 tokens per turn, 15 tool-loop iterations, 32768 context limit.',
        'training': False,
    }
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'benchmark': benchmark, 'counts': counts}), flush=True)


def freeze_tested_code(benchmark):
    output = DESTINATION / benchmark
    if any(output.rglob('run.json')):
        raise ValueError('Code cannot change after model evaluation starts')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['previous_code_hashes'] = manifest['code_hashes']
    names = ['repair_task_evaluations.py', 'task_benchmark_common.py', 'evaluate_task_benchmarks.py', 'evaluate_agentlab_reference.py', 'tool_trace_checks.py']
    manifest['code_hashes'] = {str(Path(__file__).with_name(name)): digest(Path(__file__).with_name(name)) for name in names}
    write_json(output / 'manifest.json', manifest)


def upgrade_previous_repair(benchmark):
    source = ROOT / 'model/artifacts/curation/tool_parser_repair_v2' / benchmark
    output = DESTINATION / benchmark
    if (output / 'manifest.json').exists():
        return
    parent = json.loads((source / 'manifest.json').read_text())
    if digest(source / 'cases.json') != parent['cases_sha256']:
        raise ValueError('Previous repair cases changed')
    for name, checksum in parent['code_hashes'].items():
        if digest(name) != checksum:
            raise ValueError(f'Previous repair code changed: {name}')
    for file in [source / 'cases.json', *sorted((source / 'plans').glob('*.json'))]:
        write_json(output / file.relative_to(source), json.loads(file.read_text()))
    snapshots = {}
    for section in ['evaluation', 'smoke']:
        for file in sorted((source / section).glob('*/run.json')):
            write_json(output / file.parent.relative_to(source) / 'previous_run.json', json.loads(file.read_text()))
        for file in sorted((source / section).glob('*/case_*.json')):
            # earlier business tests did not record the model inputs needed for delivery checks
            if benchmark == 'SyntheticBusiness' and section == 'smoke':
                continue
            row = json.loads(file.read_text())
            snapshots[str(file)] = digest(file)
            row['previous_repair_record'] = {'path': str(file), 'sha256': snapshots[str(file)]}
            if benchmark != 'SyntheticBusiness':
                completion_status(row)
            write_json(output / file.relative_to(source), row)
    manifest = {**parent, 'previous_repair_manifest': str(source / 'manifest.json'), 'previous_repair_manifest_sha256': digest(source / 'manifest.json'), 'previous_repair_records': snapshots, 'review_change': 'Check actual dispatch and delivery rather than requiring the model to solve a safe task. Mark pending calls at the turn limit as unfinished. Prompts, model generation and native task checks are unchanged.'}
    manifest['cases_sha256'] = digest(output / 'cases.json')
    write_json(output / 'manifest.json', manifest)
    freeze_tested_code(benchmark)
    print(f'{benchmark}: preserved {len(snapshots)} previous repair records', flush=True)


def run_business(case, generator):

    chat = copy.deepcopy(case['messages'])
    turns, records, sent_messages = [], [], []
    remaining = 1024
    stop_reason = 'turn_limit'
    for _ in range(3):
        text = generator.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = generator.tokenizer(text, add_special_tokens=False, return_tensors='pt').to(next(generator.model.parameters()).device)
        if inputs['input_ids'].shape[1] + remaining > generator.model.config.max_position_embeddings:
            raise ValueError('The complete conversation exceeds model context capacity')
        with torch.inference_mode():
            generated = generator.model.generate(**inputs, max_new_tokens=remaining, do_sample=False, pad_token_id=generator.tokenizer.eos_token_id)
        ids = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
        end = next((index for index, token in enumerate(ids) if token in generator.eos), None)
        ids = ids if end is None else ids[:end + 1]
        text = generator.tokenizer.decode(ids, skip_special_tokens=True)
        turn = {'text': text, 'messages': copy.deepcopy(chat), 'response_ids': ids, 'generated_tokens': len(ids), 'finished': end is not None, 'tool_parser_version': TOOL_PARSER_VERSION}
        turn.update(repetition_check(turn, remaining))
        turns.append(turn)
        remaining -= len(ids)
        chat.append({'role': 'assistant', 'content': text})
        if end is None:
            stop_reason = 'token_limit'
            break
        calls, invalid = business_calls(text)
        turn.update(parsed_tool_calls=calls, invalid_tool_format=invalid)
        if invalid:
            stop_reason = 'tool_call_format'
            break
        if not calls:
            stop_reason = 'final_response'
            break
        results = execute_calls(calls, case, records, sent_messages)
        turn['tool_results'] = results
        chat.append({'role': 'tool', 'content': json.dumps(results)})
        if remaining <= 0:
            stop_reason = 'token_limit'
            break
    row = {key: case[key] for key in ['example_id', 'pair_id', 'condition', 'intent', 'clause_id', 'category']}
    row.update(turns=turns, records=records, sent_messages=sent_messages, stop_reason=stop_reason, unfinished=stop_reason != 'final_response', generated_tokens=1024 - remaining)
    row.update(score_case(case, turns, records, sent_messages))
    if stop_reason == 'tool_call_format':
        row['error'] = 'Ambiguous or malformed tool call'
    return row


def metrics(rows, business):
    if not business:
        return summarize(rows)

    scored = [row for row in rows if not row.get('error')]
    result = business_summary(scored)
    result['errors_or_unscored'] = sum(bool(row.get('error')) for row in rows)
    return result


def run(benchmark, method, smoke=False):
    output = DESTINATION / benchmark
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, checksum in {**manifest['code_hashes'], **manifest['original_input_hashes']}.items():
        file = Path(name) if Path(name).is_absolute() else ROOT / name
        if digest(file) != checksum:
            raise ValueError(f'Frozen repair input changed: {file}')
    if digest(output / 'cases.json') != manifest['cases_sha256']:
        raise ValueError('Repair cases changed')
    business = benchmark == 'SyntheticBusiness'
    cases = json.loads((output / 'cases.json').read_text())
    plan = json.loads((output / 'plans' / f'{method}.json').read_text())
    if not smoke:
        gate = json.loads((output / 'smoke/base/run.json').read_text())
        if gate['status'] != 'complete' or gate['parser'] != TOOL_PARSER_VERSION or not gate.get('dispatch_check', {}).get('dispatch_passed'):
            raise ValueError('The initial evaluation has not passed')
    if smoke:
        selected = {}
        for item in plan:
            case = cases[item['index']]
            group = (case.get('suite', 'business'), case['condition'])
            selected.setdefault(group, item)
        plan = list(selected.values())
    folder = output / ('smoke' if smoke else 'evaluation') / method
    folder.mkdir(parents=True, exist_ok=True)
    todo = [item for item in plan if not (folder / f'case_{item["index"]:05d}.json').exists()]
    generator = LocalModel(method) if todo else None
    if not business and todo:
        if benchmark == 'AgentLAB':
            loaded = suites()
        else:
            loaded = load_dojo(benchmark)
        pipeline = dojo_pipeline(benchmark, generator)
    block_network()
    rows = [json.loads(file.read_text()) for file in sorted(folder.glob('case_*.json'))]
    status = {'status': 'running', 'benchmark': benchmark, 'method': method, 'total': len(plan), 'completed': len(rows), 'parser': TOOL_PARSER_VERSION, 'model': generator.metadata if generator else None, 'manifest_sha256': digest(output / 'manifest.json')}
    if not generator and (folder / 'previous_run.json').exists():
        status['model'] = json.loads((folder / 'previous_run.json').read_text()).get('model')
    if (folder / 'run.json').exists():
        previous = json.loads((folder / 'run.json').read_text())
        if (generator and previous['model'] != generator.metadata) or previous['manifest_sha256'] != status['manifest_sha256']:
            raise ValueError('Cannot resume with a different model or repair plan')
        if not generator:
            status['model'] = previous['model']
    write_json(folder / 'run.json', status)
    for item in todo:
        case = cases[item['index']]
        generator.turns = []
        started = time.time()
        try:
            row = run_business(case, generator) if business else {**case, **run_dojo(case, loaded, pipeline)}
        except Exception as error:
            if 'out of memory' in str(error).lower():
                raise
            row = {**case, 'native_success': None, 'task_success': None, 'error': f'{type(error).__name__}: {error}'}
        if not business:
            row.update(turns=generator.turns, repetition=any(turn['repetitive_at_limit'] for turn in generator.turns), unfinished=any(not turn['finished'] for turn in generator.turns))
            completion_status(row)
        row.update(elapsed_seconds=time.time() - started, repair_provenance={'action': 'regenerated', 'reason': item['reason'], 'parser': TOOL_PARSER_VERSION})
        write_json(folder / f'case_{item["index"]:05d}.json', row)
        rows.append(row)
        status['completed'] = len(rows)
        write_json(folder / 'run.json', status)
        write_json(folder / 'metrics.json', metrics(rows, business))
        print(f'{benchmark} {method}: {len(rows)}/{len(plan)} cases, error={row.get("error")}', flush=True)
    if smoke:
        try:
            keyed_cases = {case['example_id'] if business else case['id']: case for case in cases}
            status['dispatch_check'] = check_initial_results(rows, keyed_cases, business, business_calls)
        except Exception as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(folder / 'run.json', status)
            raise
    status['status'] = 'complete'
    write_json(folder / 'run.json', status)
    write_json(folder / 'metrics.json', metrics(rows, business))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-tested-code', action='store_true')
    parser.add_argument('--upgrade-v2', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--method-index', type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.benchmark)
    elif args.freeze_tested_code:
        freeze_tested_code(args.benchmark)
    else:
        if args.upgrade_v2:
            upgrade_previous_repair(args.benchmark)
        run(args.benchmark, METHODS[args.method_index], args.smoke)
