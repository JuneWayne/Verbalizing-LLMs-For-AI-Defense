"""Check portable preparation and run records with tiny files and a fake tokenizer"""

from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import json
import unittest

from scripts.curation.common import ROOT, digest, write_json, write_jsonl
from scripts.curation.preparation import prepare_lora_groups
from scripts.curation.training import run_manifest, train_reference_groups, training_helpers


class FakeTokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=False):
        condition = 'injection' if 'injected-' in messages[-1]['content'] else 'safe'
        return self.encode(condition) if tokenize else condition

    def encode(self, text, add_special_tokens=False):
        return [1, 3] if text == 'injection' else [1, 2]

    def decode(self, tokens, skip_special_tokens=True):
        return 'reference answer'


class PortableTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads((ROOT / 'scripts/curation/configs/config_reference_groups.json').read_text())
        self.config.update(model='model', data='prepared', selection_source='selections', validation_source='splits/validation.jsonl', testing_source='splits/testing.jsonl', output='run', methods=['random'], training_pairs=1, validation_pairs=1, testing_pairs=1, max_input_tokens=8, max_task_response_tokens=8)
        self.path = self.root / 'config.json'
        write_json(self.path, self.config)
        write_json(self.root / 'model/config.json', {'num_hidden_layers': 28, 'vocab_size': 10})
        (self.root / 'model/weights.safetensors').write_bytes(b'fixture model bytes')

        # Distinct source IDs and content keep the three tiny splits separate
        self.training = self.source('training', 1)
        self.validation = self.source('validation', 2)
        self.testing = self.source('testing', 3)
        write_jsonl(self.root / 'selections/random/pairs.jsonl', [self.training])
        write_jsonl(self.root / 'splits/validation.jsonl', [self.validation])
        write_jsonl(self.root / 'splits/testing.jsonl', [self.testing])
        for name, source in [('random', self.training), ('validation', self.validation)]:
            token_path = self.root / ('lora_validation.jsonl' if name == 'validation' else 'lora_training_odile/random.jsonl')
            row = {'pair_id': source['pair_id'], 'prefix_ids': {'safe': [1, 2], 'injection': [1, 3]}, 'completion_ids': {'safe': [4, 9], 'injection': [5, 9]}}
            write_jsonl(token_path, [row])
            metadata = {'sha256': digest(token_path)} if name == 'validation' else {'training_sha256': digest(token_path), 'selection_sha256': digest(self.root / 'selections/random/pairs.jsonl')}
            write_json(token_path.with_suffix('.json'), metadata)

        # These files stand in for source code when testing manifest fingerprints
        for name in run_manifest.TRAINING_CODE_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# fixture source\n')
        for module in [run_manifest, prepare_lora_groups, training_helpers]:
            context = patch.object(module, 'ROOT', self.root)
            context.start()
            self.addCleanup(context.stop)

    def source(self, pair_id, source_id):
        return {'pair_id': pair_id, 'user_prompt_id': source_id, 'user_instruction': 'Summarize', 'clean_data': f'clean-{pair_id}', 'injected_data': f'injected-{pair_id}', 'witness': 'blue'}

    def prepare(self):
        with patch.object(prepare_lora_groups.AutoTokenizer, 'from_pretrained', return_value=FakeTokenizer()):
            return prepare_lora_groups.prepare(self.path)

    def test_preparation_needs_no_cluster_or_approval_files(self):
        report = self.prepare()
        self.assertEqual(report['pair_counts'], {'random': 1, 'validation': 1})
        self.assertFalse(report['training_started'])
        self.assertFalse(list(self.root.rglob('approval.json')))
        self.assertFalse((self.root / 'bash').exists())

    def test_manifest_is_automatic_and_reusable_for_unchanged_inputs(self):
        self.prepare()
        record = run_manifest.record_manifest(self.path, run_manifest.read_config(self.path))
        self.assertEqual(record, run_manifest.record_manifest(self.path, self.config))
        self.assertEqual(record['model_hashes']['weights.safetensors'], digest(self.root / 'model/weights.safetensors'))
        self.assertFalse(list(self.root.rglob('approval.json')))

    def test_changed_weights_require_another_output_folder(self):
        self.prepare()
        run_manifest.record_manifest(self.path, self.config)
        (self.root / 'model/weights.safetensors').write_bytes(b'changed fixture bytes')
        with self.assertRaisesRegex(ValueError, 'different inputs or settings'):
            run_manifest.record_manifest(self.path, self.config)

    def test_overlength_reference_is_rejected_before_export(self):
        self.config['max_task_response_tokens'] = 1
        write_json(self.path, self.config)
        with self.assertRaisesRegex(ValueError, 'task response'):
            self.prepare()
        self.assertFalse((self.root / 'prepared/training/random.jsonl').exists())

    def test_source_overlap_is_still_rejected(self):
        self.validation['user_prompt_id'] = self.training['user_prompt_id']
        write_jsonl(self.root / 'splits/validation.jsonl', [self.validation])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.prepare()

    def test_check_only_never_loads_a_model_or_starts_training(self):
        self.prepare()
        command = ['train_reference_groups', '--config', str(self.path), '--check-only']
        with patch('sys.argv', command), patch.object(training_helpers.AutoTokenizer, 'from_pretrained', return_value=FakeTokenizer()), patch.object(train_reference_groups, 'load_model') as load_model, patch.object(train_reference_groups, 'train') as train, patch.object(train_reference_groups, 'record_manifest') as manifest, redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as stopped:
                train_reference_groups.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertFalse(json.loads(output.getvalue())['training_started'])
        load_model.assert_not_called()
        train.assert_not_called()
        manifest.assert_not_called()


if __name__ == '__main__':
    unittest.main()
