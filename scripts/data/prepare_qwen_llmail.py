"""Prepare the existing LLMail test emails for the unchanged Qwen classifiers."""

from pathlib import Path
import json

from transformers import AutoTokenizer
import pandas as pd

from scripts.analysis.experiment_files import file_hash, saved_path
from scripts.data.prepare_llmail_probe_transfer import normalized_text


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/llmail/curated/probe_transfer_antares_1b.csv'
OUTPUT = ROOT / 'data/llmail/curated/qwen_detector_test_v1.csv'
MODELS = ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-7b']


def main():
    """Reuse the Qwen prompt and selected detectors with the fixed LLMail testing emails."""
    source_manifest = json.loads(SOURCE.with_suffix('.json').read_text())
    if file_hash(SOURCE) != source_manifest['dataset sha256']:
        raise ValueError('The original LLMail test emails changed.')
    data = pd.read_csv(SOURCE, engine='python', keep_default_na=False)
    if len(data) != 1110 or data['condition'].value_counts().to_dict() != {'injection': 1000, 'safe': 110}:
        raise ValueError('Use all original 1,110 LLMail emails, without resampling.')
    if data['email_id'].duplicated().any() or not data['condition'].equals(data['label'].map({0: 'safe', 1: 'injection'})):
        raise ValueError('The LLMail IDs or labels are invalid.')

    cami_path = ROOT / 'data/cami/curated/qwen_family_pairs_v2.csv'
    cami_manifest = json.loads(cami_path.with_suffix('.json').read_text())
    signature = cami_manifest['shared prompt signature']
    if file_hash(cami_path) != cami_manifest['dataset sha256']:
        raise ValueError('The approved Qwen CAMI inputs changed.')
    prompt_path = ROOT / 'model/artifacts/antares-1b/probe_inputs/run.json'
    if file_hash(prompt_path) != cami_manifest['prompt source sha256']:
        raise ValueError('The original classification prompt changed.')
    prompt = json.loads(prompt_path.read_text())

    # Check both full source emails and the exact text used to fit the probes and lenses.
    training_paths = [ROOT / 'data/cami_real_pairs.csv', cami_path, ROOT / 'data/llmail/curated/jlens_fit_100_balanced_under_128.csv']
    excluded = set()
    for path in training_paths:
        frame = pd.read_csv(path, engine='python', keep_default_na=False)
        column = 'email text' if path == cami_path else 'text'
        excluded.update(frame[column].map(normalized_text))
    if data['text'].map(normalized_text).isin(excluded).any():
        raise ValueError('A test email overlaps the probe or lens source data.')

    model_sources, tokenized = {}, []
    for model in MODELS:
        base = ROOT / 'model/artifacts' / model / 'family_comparison_v1'
        frozen = json.loads((base / 'frozen_inputs/run.json').read_text())
        fit = json.loads((base / 'reader_comparison/run.json').read_text())
        if frozen['dataset sha256'] != cami_manifest['dataset sha256'] or fit['frozen run sha256'] != file_hash(base / 'frozen_inputs/run.json'):
            raise ValueError('A Qwen probe uses different CAMI inputs.')
        model_path = saved_path(frozen['model path'])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        marker = '\0EMAIL\0'
        messages = [{'role': 'system', 'content': prompt['system prompt']}, {'role': 'user', 'content': prompt['classification template'].format(email=marker)}]
        shell = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + 'Classification label:'
        head, tail = shell.split(marker)
        actual = {
            'tokenizer sha256': file_hash(model_path / 'tokenizer.json'),
            'head ids': tokenizer.encode(head, add_special_tokens=True),
            'tail ids': tokenizer.encode(tail, add_special_tokens=False),
            'label ids': {label: tokenizer.encode(' ' + label, add_special_tokens=False) for label in ['injection', 'safe']},
        }
        if actual != signature or '<think>' in shell:
            raise ValueError('The model no longer uses the original Qwen classification template.')
        email_ids = [tokenizer.encode(text, add_special_tokens=False) for text in data['text']]
        input_ids = [signature['head ids'] + ids + signature['tail ids'] for ids in email_ids]
        if tokenized and input_ids != tokenized:
            raise ValueError('Qwen sizes tokenize these emails differently.')
        tokenized = input_ids
        model_sources[model] = {
            'frozen run sha256': file_hash(base / 'frozen_inputs/run.json'),
            'fit run sha256': file_hash(base / 'reader_comparison/run.json'),
            'selection sha256': file_hash(base / 'reader_comparison/selected_readers.json'),
        }

    prepared = data.rename(columns={'email_id': 'example_id', 'email tokens': 'Antares email tokens'}).copy()
    prepared['email tokens'] = [len(ids) for ids in email_ids]
    prepared['within CAMI email limit'] = prepared['email tokens'].le(cami_manifest['max email tokens'])
    prepared['prompt tokens'] = [len(ids) for ids in tokenized]
    prepared['input token ids'] = [json.dumps(ids) for ids in tokenized]
    prepared['split'] = 'test'
    settings = {
        'source path': str(SOURCE),
        'source sha256': file_hash(SOURCE),
        'CAMI dataset sha256': file_hash(cami_path),
        'shared prompt signature': signature,
        'model': model_sources,
        'training source sha256': {str(path): file_hash(path) for path in training_paths},
        'examples': len(prepared),
        'class counts': prepared['condition'].value_counts().to_dict(),
        'max email tokens': int(prepared['email tokens'].max()),
        'above CAMI email limit': int((~prepared['within CAMI email limit']).sum()),
        'input policy': 'all original complete LLMail emails; no truncation or new selection; secondary comparison excludes emails above 512 Qwen tokens',
        'pairs': False,
        'fitting': False,
        'generator sha256': file_hash(Path(__file__)),
    }
    if OUTPUT.exists():
        saved = json.loads(OUTPUT.with_suffix('.json').read_text())
        if saved != {**settings, 'dataset sha256': file_hash(OUTPUT)}:
            raise ValueError('Existing prepared inputs use different sources or settings.')
        pd.testing.assert_frame_equal(pd.read_csv(OUTPUT, engine='python', keep_default_na=False), prepared)
        print(f'Existing LLMail inputs verified: {OUTPUT}')
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT, index=False)
    settings['dataset sha256'] = file_hash(OUTPUT)
    OUTPUT.with_suffix('.json').write_text(json.dumps(settings, indent=2))
    print(prepared.groupby('condition')['email tokens'].agg(['size', 'min', 'max']).to_string())
    print(f'Saved complete test emails: {OUTPUT}. No fitting started.')


if __name__ == '__main__':
    main()
