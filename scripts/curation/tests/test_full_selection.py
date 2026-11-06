"""Check the full-pool split and chunked scoring without running a model"""

from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

from sklearn.linear_model import LogisticRegression
import numpy as np
import torch

from scripts.curation.common import load_pairs, read_jsonl, settings
from scripts.curation.selection.data_curation_selection import lens_differences, score_lens, score_probe, state_batches


class FullSelectionChecks(unittest.TestCase):
    def test_all_training_rows_accounted_for(self):
        config, output = settings()
        manifest = json.loads((output / "manifest.json").read_text())
        audit = read_jsonl(output / "pool_audit.jsonl")
        self.assertEqual(len(audit), 100000)
        self.assertEqual(len({row["pair_id"] for row in audit}), 100000)
        self.assertEqual(sum(manifest["training_row_use"].values()), 100000)
        self.assertEqual(manifest["split_pairs"]["selector_training"], 1500)
        self.assertEqual(manifest["split_pairs"]["validation"], 500)
        self.assertEqual(manifest["split_pairs"]["testing"], 10000)
        self.assertEqual(config["max_new_tokens"], 2048)

    def test_sources_stay_separate(self):
        splits = {name: load_pairs(name) for name in ["selector_training", "validation", "candidates", "testing"]}
        for index, first in enumerate(splits):
            for second in list(splits)[index + 1:]:
                for key in ["user_prompt_id", "clean_data"]:
                    self.assertFalse({row[key] for row in splits[first]} & {row[key] for row in splits[second]})

    def test_candidates_keep_multiple_injection_variants(self):
        rows = load_pairs("candidates")
        self.assertGreater(len(rows), len({row["user_prompt_id"] for row in rows}))

    def test_stream_checks_order_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "states/candidates"
            folder.mkdir(parents=True)
            hidden = np.arange(12).reshape(2, 2, 3).astype(np.float16)
            np.savez(folder / "000000.npz", states=hidden, labels=[0, 1], pair_ids=["a", "a"])
            with patch("scripts.curation.selection.data_curation_selection.settings", return_value=({}, Path(directory))), patch("scripts.curation.selection.data_curation_selection.load_pairs", return_value=[{"pair_id": "a"}]):
                batches = list(state_batches("candidates"))
                np.testing.assert_array_equal(batches[0][0], hidden)
            with patch("scripts.curation.selection.data_curation_selection.settings", return_value=({}, Path(directory))), patch("scripts.curation.selection.data_curation_selection.load_pairs", return_value=[{"pair_id": "b"}]):
                with self.assertRaises(ValueError):
                    list(state_batches("candidates"))
            with patch("scripts.curation.selection.data_curation_selection.settings", return_value=({}, Path(directory))), patch("scripts.curation.selection.data_curation_selection.load_pairs", return_value=[{"pair_id": "a"}, {"pair_id": "b"}]):
                with self.assertRaises(ValueError):
                    list(state_batches("candidates"))

    def test_chunked_lens_matches_single_layer_calculation(self):
        rng = np.random.default_rng(42)
        hidden = rng.normal(size=(4, 3, 8)).astype(np.float32)
        matrix = torch.tensor(rng.normal(size=(8, 8)), dtype=torch.float32)
        readout = {"norm": torch.ones(8), "unembedding": torch.tensor(rng.normal(size=(2, 8)), dtype=torch.float32), "eps": 1e-6}
        expected = lens_differences(hidden[:, 1:2], {0: matrix}, readout)[:, 0]
        batches = [(hidden[:2], np.array([0, 1]), ["a", "a"]), (hidden[2:], np.array([0, 1]), ["b", "b"])]
        with patch("scripts.curation.selection.data_curation_selection.state_batches", return_value=iter(batches)), patch("scripts.curation.selection.data_curation_selection.save_probability_scores") as save:
            score_lens("example", matrix, readout, 1)
            np.testing.assert_array_equal(save.call_args.args[4], expected)

    def test_chunked_probe_uses_selected_layer(self):

        hidden = np.random.default_rng(42).normal(size=(8, 3, 4)).astype(np.float32)
        labels = np.tile([0, 1], 4)
        probe = LogisticRegression().fit(hidden[:, 2], labels)
        ids = np.repeat(["a", "b", "c", "d"], 2)
        batches = [(hidden[:4], labels[:4], ids[:4]), (hidden[4:], labels[4:], ids[4:])]
        with patch("scripts.curation.selection.data_curation_selection.state_batches", return_value=iter(batches)), patch("scripts.curation.selection.data_curation_selection.save_probability_scores") as save:
            score_probe("example", probe, 2)
            np.testing.assert_allclose(save.call_args.args[3], probe.predict_proba(hidden[:, 2])[:, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
