"""Regression checks for missed calls, missing observations, and unfinished runs"""

import copy
import json
import unittest

from scripts.curation.evaluation.repair_task_evaluations import business_calls
from scripts.curation.evaluation.synthetic_business_rules import execute_calls
from scripts.curation.evaluation.task_benchmark_common import TOOL_PARSER_VERSION
from scripts.curation.evaluation.tool_trace_checks import (
    check_agent_trace,
    check_business_trace,
    check_initial_results,
    completion_status,
)


def example_trace(executions=1, pending=False):
    trace, turns = [], []
    previous = []
    for index in range(executions + 1):
        has_call = index < executions or pending
        call = {'function': 'search', 'args': {'query': 'watch'}, 'id': f'call_{index}'}
        text = 'I will search.\n<tool_call>{"name":"search","arguments":{"query":"watch"}}</tool_call>' if has_call else 'Would you like to buy the watch?'
        turns.append({'text': text, 'finished': True, 'messages': copy.deepcopy(previous), 'tool_parser_version': TOOL_PARSER_VERSION})
        trace.append({'role': 'assistant', 'content': [], 'tool_calls': [call] if has_call else []})
        if index < executions:
            trace.append({'role': 'tool', 'content': [{'type': 'text', 'content': 'A watch is available'}], 'tool_call_id': call['id'], 'tool_call': copy.deepcopy(call), 'error': None})
            previous = [{'role': 'tool', 'content': 'A watch is available'}]
    return {'id': 'safe_example', 'condition': 'safe', 'turns': turns, 'trace': trace, 'task_success': False, 'native_success': False}


class TraceTests(unittest.TestCase):
    def test_missing_response_is_not_finished(self):
        row = completion_status({'trace': [], 'turns': []})
        self.assertEqual(row['stop_reason'], 'evaluation_error')
        self.assertTrue(row['unfinished'])

    def test_model_failure_is_not_a_dispatch_failure(self):
        row = example_trace()
        result = check_initial_results([row], {})
        self.assertTrue(result['dispatch_passed'])
        self.assertEqual(result['safe_tasks_completed'], 0)
        self.assertIsNotNone(result['warning'])
        self.assertFalse(row['task_success'])

    def test_missing_request_is_detected(self):
        row = example_trace()
        row['trace'][0]['tool_calls'] = []
        with self.assertRaisesRegex(ValueError, 'omitted or changed'):
            check_agent_trace(row)

    def test_missing_execution_is_detected(self):
        row = example_trace()
        del row['trace'][1]
        with self.assertRaisesRegex(ValueError, 'matching execution'):
            check_agent_trace(row)

    def test_wrong_tool_arguments_are_detected(self):
        row = example_trace()
        row['trace'][1]['tool_call']['args']['query'] = 'another product'
        with self.assertRaisesRegex(ValueError, 'arguments differ'):
            check_agent_trace(row)

    def test_wrong_result_id_is_detected(self):
        row = example_trace()
        row['trace'][1]['tool_call_id'] = 'another_call'
        with self.assertRaisesRegex(ValueError, 'different request'):
            check_agent_trace(row)

    def test_missing_tool_observation_is_detected(self):
        row = example_trace()
        row['turns'][1]['messages'] = []
        with self.assertRaisesRegex(ValueError, 'did not receive'):
            check_agent_trace(row)

    def test_native_turn_limit_is_reported_as_unfinished(self):
        row = completion_status(example_trace(executions=15, pending=True))
        self.assertEqual(row['stop_reason'], 'tool_turn_limit')
        self.assertTrue(row['unfinished'])
        self.assertEqual(check_agent_trace(row)['executed_tools'], 15)

    def test_early_missing_result_is_not_excused_as_a_turn_limit(self):
        row = completion_status(example_trace(executions=1, pending=True))
        self.assertEqual(row['stop_reason'], 'missing_tool_result')
        with self.assertRaisesRegex(ValueError, 'matching execution'):
            check_agent_trace(row)

    def test_no_tool_execution_blocks_the_initial_check(self):
        row = example_trace(executions=0)
        with self.assertRaisesRegex(ValueError, 'No tools executed'):
            check_initial_results([row], {})

    def test_tool_error_reaches_the_model(self):
        row = example_trace()
        row['trace'][1]['error'] = 'Product not found'
        row['turns'][1]['messages'][-1]['content'] = 'Product not found'
        self.assertEqual(check_agent_trace(row)['executed_tools'], 1)


class BusinessTraceTests(unittest.TestCase):
    def setUp(self):
        self.case = {'case_id': 'CASE-1', 'intent': 'DECOMMISSION'}
        self.call = {'name': 'submit_request', 'arguments': {'case_id': 'CASE-1', 'business_task': 'DECOMMISSION', 'summary': 'Prepare the request', 'parameters': {}}}
        records, messages = [], []
        results = execute_calls([self.call], self.case, records, messages)
        first = {'text': 'I will submit the request.\n<tool_call>' + json.dumps(self.call) + '</tool_call>', 'finished': True, 'tool_parser_version': TOOL_PARSER_VERSION, 'tool_results': results, 'messages': []}
        second = {'text': 'The request was submitted', 'finished': True, 'tool_parser_version': TOOL_PARSER_VERSION, 'messages': [{'role': 'tool', 'content': json.dumps(results)}]}
        self.row = {'turns': [first, second], 'records': records, 'sent_messages': messages, 'stop_reason': 'final_response'}

    def test_dispatch_and_delivery(self):
        self.assertEqual(check_business_trace(self.row, self.case, business_calls)['executed_tools'], 1)

    def test_missing_delivery(self):
        self.row['turns'][1]['messages'] = []
        with self.assertRaisesRegex(ValueError, 'did not receive'):
            check_business_trace(self.row, self.case, business_calls)

    def test_changed_record(self):
        self.row['records'] = []
        with self.assertRaisesRegex(ValueError, 'Saved business actions'):
            check_business_trace(self.row, self.case, business_calls)

    def test_false_finished_status(self):
        self.row['turns'].pop()
        with self.assertRaisesRegex(ValueError, 'subsequent model turn'):
            check_business_trace(self.row, self.case, business_calls)

    def test_tool_execution_at_turn_limit(self):
        self.row['turns'].pop()
        self.row['stop_reason'] = 'turn_limit'
        self.assertEqual(check_business_trace(self.row, self.case, business_calls)['executed_tools'], 1)


if __name__ == '__main__':
    unittest.main()
