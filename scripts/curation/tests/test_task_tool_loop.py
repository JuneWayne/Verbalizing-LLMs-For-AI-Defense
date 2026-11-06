"""CPU-only tool-loop checks with fixed responses and the upstream benchmark runtime"""

import unittest

from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime, make_function

from scripts.curation.evaluation.evaluate_task_benchmarks import dojo_pipeline
from scripts.curation.evaluation.task_benchmark_common import TOOL_PARSER_VERSION


class SavedResponses:
    def __init__(self, texts, finished=True):
        self.texts = iter(texts)
        self.finished = finished
        self.turns = []
        self.received = []

    def generate(self, messages, tools=None):
        self.received.append(messages)
        result = {'text': next(self.texts), 'finished': self.finished}
        self.turns.append(result)
        return result


class ToolLoopTests(unittest.TestCase):
    def setUp(self):
        self.actions = []

        def search_product(product_name: str, category: str) -> str:
            """Search the simulated catalog.

            :param product_name: Product requested by the user
            :param category: Product category
            """
            self.actions.append((product_name, category))
            return 'A smart watch is available'

        self.runtime = FunctionsRuntime([make_function(search_product)])
        self.call = '<tool_call>{"name":"search_product","arguments":{"product_name":"smart watch","category":"Electronics"}}</tool_call>'

    def test_explanation_call_tool_result_final_answer(self):
        generator = SavedResponses(['I will search for a smart watch.\n' + self.call, 'A smart watch is available.'])
        result = dojo_pipeline('AgentDojo', generator).query('Find a smart watch', self.runtime, EmptyEnv(), [], {})
        self.assertEqual(self.actions, [('smart watch', 'Electronics')])
        self.assertEqual(len(generator.turns), 2)
        self.assertEqual(generator.received[1][-1]['role'], 'tool')
        self.assertIn('A smart watch is available', generator.received[1][-1]['content'])
        self.assertEqual(result[3][-1]['tool_calls'], [])
        self.assertEqual(generator.turns[0]['tool_parser_version'], TOOL_PARSER_VERSION)

    def test_unknown_tool_returns_runtime_error(self):
        generator = SavedResponses(['I will search.\n<tool_call>{"name":"missing_tool","arguments":{}}</tool_call>', 'The requested tool is unavailable.'])
        dojo_pipeline('AgentDojo', generator).query('Find a smart watch', self.runtime, EmptyEnv(), [], {})
        self.assertEqual(self.actions, [])
        self.assertIn('Invalid tool', generator.received[1][-1]['content'])

    def test_malformed_call_raises_instead_of_finishing_as_safe(self):
        generator = SavedResponses(['I will search.\n<tool_call>{"name":'])
        with self.assertRaisesRegex(ValueError, 'tool_call_format'):
            dojo_pipeline('AgentDojo', generator).query('Find a smart watch', self.runtime, EmptyEnv(), [], {})
        self.assertEqual(self.actions, [])

    def test_unfinished_response_never_executes(self):
        generator = SavedResponses([self.call], finished=False)
        with self.assertRaisesRegex(ValueError, 'tool_call_format'):
            dojo_pipeline('AgentDojo', generator).query('Find a smart watch', self.runtime, EmptyEnv(), [], {})
        self.assertEqual(self.actions, [])


if __name__ == '__main__':
    unittest.main()
