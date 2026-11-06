"""Check tool dispatch separately from the model's ability to complete a task"""

import ast
import copy
import json

from scripts.curation.evaluation.synthetic_business_rules import execute_calls
from scripts.curation.evaluation.task_benchmark_common import TOOL_PARSER_VERSION, parse_tool_calls


def normalized_arguments(arguments):
    result = copy.deepcopy(arguments)
    # the upstream executor converts strings containing Python lists into lists
    for name, value in result.items():
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                result[name] = parsed
    return result


def completion_status(row):
    if row.get('error') or not row.get('turns'):
        reason = 'evaluation_error'
    elif any(not turn.get('finished', False) for turn in row.get('turns', [])):
        reason = 'token_limit'
    else:
        trace = row.get('trace', [])
        pending = trace and trace[-1]['role'] == 'assistant' and trace[-1].get('tool_calls')
        reason = 'tool_turn_limit' if pending and len(row.get('turns', [])) == 16 else 'missing_tool_result' if pending else 'final_response'
    row.update(stop_reason=reason, unfinished=reason != 'final_response')
    return row


def check_agent_trace(row):
    if row.get('error'):
        raise ValueError(row['error'])
    trace = row.get('trace', [])
    turns = row.get('turns', [])
    positions = [index for index, message in enumerate(trace) if message['role'] == 'assistant']
    if not turns or len(positions) != len(turns):
        raise ValueError('Model turns do not match the saved conversation')
    executed = 0
    for number, position in enumerate(positions):
        turn = turns[number]
        if turn.get('tool_parser_version') != TOOL_PARSER_VERSION:
            raise ValueError('The initial test used a different parser')
        calls, invalid = parse_tool_calls(turn['text'], finished=turn['finished'])
        if invalid:
            raise ValueError('Ambiguous or incomplete call in the initial test')
        actual = trace[position].get('tool_calls') or []
        expected_calls = [(call['name'], normalized_arguments(call['arguments'])) for call in calls]
        actual_calls = [(call['function'], normalized_arguments(call['args'])) for call in actual]
        if expected_calls != actual_calls:
            raise ValueError('A parsed tool call was omitted or changed')
        results = []
        for message in trace[position + 1:]:
            if message['role'] != 'tool':
                break
            results.append(message)
        if len(results) != len(calls):
            # the published loop stops after 15 executions and can leave the last request pending
            if number == 15 and len(turns) == 16 and not results and calls:
                continue
            raise ValueError('A tool request has no matching execution result')
        for call, result in zip(actual, results):
            if result.get('tool_call_id') != call.get('id'):
                raise ValueError('Tool result belongs to a different request')
            dispatched = result.get('tool_call', {})
            if dispatched.get('function') != call['function'] or normalized_arguments(dispatched.get('args', {})) != normalized_arguments(call['args']):
                raise ValueError('The executed tool or arguments differ from the request')
        executed += len(results)
        if results:
            if number + 1 >= len(turns):
                raise ValueError('Tool results were not followed by another model turn')
            expected_input = [{'role': 'tool', 'content': result.get('error') or '\n'.join(block['content'] for block in result['content'] if block.get('content') is not None)} for result in results]
            actual_input = turns[number + 1].get('messages', [])[-len(results):]
            if expected_input != actual_input:
                raise ValueError('The next model turn did not receive the tool results')
    return {'executed_tools': executed, 'verified_model_turns': len(turns)}


def check_business_trace(row, case, parser):

    if row.get('error'):
        raise ValueError(row['error'])
    records, sent_messages = [], []
    turns = row.get('turns', [])
    executed = 0
    if not turns:
        raise ValueError('No model response in the business test')
    for index, turn in enumerate(turns):
        if turn.get('tool_parser_version') != TOOL_PARSER_VERSION:
            raise ValueError('The business test used a different parser')
        calls, invalid = parser(turn['text'], turn['finished'])
        if invalid:
            raise ValueError('Ambiguous or incomplete business call')
        expected = execute_calls(calls, case, records, sent_messages) if calls else []
        if expected != turn.get('tool_results', []):
            raise ValueError('Business tool execution differs from the parsed request')
        executed += len(calls)
        if calls and index + 1 < len(turns):
            messages = turns[index + 1].get('messages', [])
            if not messages or messages[-1] != {'role': 'tool', 'content': json.dumps(expected)}:
                raise ValueError('The next business response did not receive the tool results')
        elif calls and row.get('stop_reason') not in ['turn_limit', 'token_limit']:
            raise ValueError('Business tool results did not reach a subsequent model turn')
    if records != row.get('records', []) or sent_messages != row.get('sent_messages', []):
        raise ValueError('Saved business actions differ from executed actions')
    return {'executed_tools': executed, 'verified_model_turns': len(turns)}


def check_initial_results(rows, cases, business=False, parser=None):
    if not rows or any(row.get('error') for row in rows):
        raise ValueError('Initial evaluation contains errors or no results')
    checks = []
    for row in rows:
        checks.append(check_business_trace(row, cases[row['example_id']], parser) if business else check_agent_trace(row))
    if not sum(item['executed_tools'] for item in checks):
        raise ValueError('No tools executed in the initial evaluation')
    safe = [row for row in rows if row['condition'] == 'safe']
    completed = sum(bool(row.get('valid_business_record_submitted') if business else row.get('task_success')) for row in safe)
    # task failure is a model result; the checks above establish whether the evaluator works
    return {'dispatch_passed': True, 'safe_cases': len(safe), 'safe_tasks_completed' if not business else 'safe_records_submitted': completed, 'warning': 'No safe task completed; interpret ASR alongside task completion' if not business and not completed else None, 'executed_tools': sum(item['executed_tools'] for item in checks)}
