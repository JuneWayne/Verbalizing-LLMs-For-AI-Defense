"""Check extracted data-validation functions without fitting a model"""

from unittest.mock import Mock
import copy
import hashlib
import unittest

from scripts.curation.training.training_helpers import check_prompt_tokens, check_rows, check_sources, check_splits


class TrainingDataChecks(unittest.TestCase):
    def setUp(self):
        self.config = {'max_input_tokens': 8, 'max_task_response_tokens': 8, 'system_prompt': 'neutral'}
        self.source = {'pair_id': 'a', 'user_prompt_id': 7, 'clean_data': 'original content', 'injected_data': 'inserted instruction', 'user_instruction': 'summarize'}
        self.row = {'pair_id': 'a', 'source_id': '7', 'clean_data_sha256': hashlib.sha256(b'original content').hexdigest(), 'prefix_ids': {'safe': [1, 2], 'injection': [1, 3]}, 'completion_ids': {'safe': [4, 9], 'injection': [5, 9]}, 'task_response_ids': [4, 9]}

    def test_complete_reference_with_end_token_is_accepted(self):
        check_rows([self.row], 1, self.config, 9, 10)
        check_sources([self.row], [self.source], exact=True)

    def test_missing_end_token_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['task_response_ids'] = [4, 5]
        with self.assertRaises(ValueError):
            check_rows([row], 1, self.config, 9, 10)

    def test_invalid_token_id_is_rejected(self):
        row = copy.deepcopy(self.row)
        row['prefix_ids']['injection'] = [11]
        with self.assertRaises(ValueError):
            check_rows([row], 1, self.config, 9, 10)

    def test_duplicate_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            check_rows([self.row, self.row], 2, self.config, 9, 10)

    def test_shared_original_content_is_rejected_across_splits(self):
        other = {**self.row, 'pair_id': 'b', 'source_id': '8'}
        with self.assertRaises(ValueError):
            check_splits([self.row], [other])

    def test_changed_source_metadata_is_rejected(self):
        row = {**self.row, 'source_id': '8'}
        with self.assertRaises(ValueError):
            check_sources([row], [self.source], exact=True)

    def test_prompt_prefix_must_match_prepared_tokens(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1, 2], [1, 3]]
        check_prompt_tokens([self.source], [self.row], tokenizer, self.config)
        tokenizer.encode.side_effect = [[1, 2], [1, 8]]
        with self.assertRaises(ValueError):
            check_prompt_tokens([self.source], [self.row], tokenizer, self.config)


if __name__ == '__main__':
    unittest.main()
