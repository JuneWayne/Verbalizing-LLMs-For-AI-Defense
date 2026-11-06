"""Check checkpoint selection and the complete reference-answer loss without fitting"""

from types import SimpleNamespace
import json
import unittest

import torch
import torch.nn.functional as F

from scripts.curation.common import ROOT
from scripts.curation.training.reference_loss import task_response_loss
from scripts.curation.training.train_reference_groups import best_checkpoint
from scripts.curation.training.training_helpers import learning_rate


class ReferenceTests(unittest.TestCase):
    def test_checkpoint_uses_validation_not_last_epoch(self):
        rows = [{'epoch': 0, 'loss': 0.1}, {'epoch': 1, 'loss': 0.5}, {'epoch': 2, 'loss': 0.8}]
        self.assertEqual(best_checkpoint(rows)['epoch'], 1)

    def test_tie_uses_earlier_epoch(self):
        self.assertEqual(best_checkpoint([{'epoch': 3, 'loss': 1}, {'epoch': 1, 'loss': 1}])['epoch'], 1)

    def test_full_answer_and_end_token_are_scored(self):
        target = [1] * 60 + [2]
        logits = torch.zeros(1, len(target) + 1, 3)
        logits[0, 60, 2] = 2
        class FakeModel:
            def __call__(self, input_ids, use_cache, logits_to_keep):
                self.ids = input_ids.tolist()[0]
                self.keep = logits_to_keep
                return SimpleNamespace(logits=logits)
        model = FakeModel()
        loss = task_response_loss(model, {'prefix_ids': {'injection': [0, 0]}, 'task_response_ids': target}, 'cpu')
        self.assertEqual(model.ids, [0, 0] + target)
        self.assertEqual(model.keep, len(target) + 1)
        self.assertAlmostEqual(loss.item(), F.cross_entropy(logits[0, :-1], torch.tensor(target)).item())

    def test_approved_schedule(self):
        config = json.loads((ROOT / 'scripts/curation/configs/config_reference_groups.json').read_text())
        self.assertEqual(config['training_pairs'] * config['epochs'], 1280)
        self.assertAlmostEqual(learning_rate(128, config), 5e-5)
        self.assertAlmostEqual(learning_rate(1280, config), 5e-6)
        self.assertFalse(config['separation_loss_enabled'])


if __name__ == '__main__':
    unittest.main()
