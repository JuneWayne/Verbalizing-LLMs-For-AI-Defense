"""Check parsing failures found in the first simulated business evaluation"""

import json
import unittest

from scripts.curation.evaluation.evaluate_reference_transfer import parse_business_calls
from scripts.curation.evaluation.synthetic_business_rules import FIELD_GOALS, execute_calls, score_case


class TransferEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.call = {'name': 'submit_request', 'arguments': {'case_id': 'CASE-1', 'business_task': 'DECOMMISSION', 'summary': 'Prepare the device request', 'parameters': {}}}
        self.text = json.dumps(self.call)

    def test_missing_closing_marker(self):
        self.assertEqual(parse_business_calls('<tool_call>' + self.text), [self.call])

    def test_missing_marker_with_final_prose(self):
        self.assertEqual(parse_business_calls('<tool_call>' + self.text + '\n\nThe request was submitted.'), [self.call])

    def test_closed_marker_with_final_prose(self):
        self.assertEqual(parse_business_calls('<tool_call>' + self.text + '</tool_call>\nDone'), [self.call])

    def test_multiple_explicit_calls(self):
        text = '<tool_call>' + self.text + '</tool_call>'
        self.assertEqual(parse_business_calls(text + '\n' + text), [self.call, self.call])

    def test_prose_and_quoted_examples_stay_text(self):
        for prefix in ['An example is ', 'Do not execute ', '> ', '```json\n']:
            self.assertEqual(parse_business_calls(prefix + '<tool_call>' + self.text), [])

    def test_incomplete_json_never_executes(self):
        self.assertEqual(parse_business_calls('<tool_call>' + self.text[:-1]), [])

    def test_unmarked_json_must_be_whole_response(self):
        self.assertEqual(parse_business_calls(self.text), [self.call])
        self.assertEqual(parse_business_calls(self.text + ' example only'), [])

    def test_unknown_tools_and_wrong_arguments(self):
        self.assertEqual(parse_business_calls('<tool_call>{"name":"shell","arguments":{}}'), [])
        self.assertEqual(parse_business_calls('<tool_call>{"name":"submit_request","arguments":[]}'), [])

    def test_recovered_call_reaches_simulated_record(self):
        case = {'case_id': 'CASE-1', 'intent': 'DECOMMISSION', 'category': 'request_field_change', 'field_goals': FIELD_GOALS[260], 'secrets': {}, 'requested_secret': None}
        self.call['arguments']['parameters'] = dict(FIELD_GOALS[260])
        text = '<tool_call>' + json.dumps(self.call)
        records, messages = [], []
        execute_calls(parse_business_calls(text), case, records, messages)
        result = score_case(case, [{'text': text, 'repetitive_at_limit': False}], records, messages)
        self.assertTrue(result['valid_business_record_submitted'])
        self.assertTrue(result['attacker_goal_reached'])


if __name__ == '__main__':
    unittest.main()
