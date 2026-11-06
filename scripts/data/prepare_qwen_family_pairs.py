"""Prepare one inspectable, length-equalized dataset for all three Qwen sizes."""

from pathlib import Path
import hashlib
import json
import re

from transformers import AutoTokenizer
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'data/cami/curated/qwen_family_pairs_v2.csv'
FOLDERS = ['Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-7B-Instruct']
SOURCE = ROOT / 'data/cami_real_pairs.csv'
PROMPT_SOURCE = ROOT / 'model/artifacts/antares-1b/probe_inputs/run.json'
MAX_EMAIL_TOKENS = 512


def main():
    """Prepare shared Qwen inputs, preserving pairs and complete inserted instructions."""
    if OUTPUT.exists():
        manifest = json.loads(OUTPUT.with_suffix('.json').read_text())
        if hashlib.sha256(OUTPUT.read_bytes()).hexdigest() != manifest['dataset sha256']:
            raise ValueError('The prepared family dataset changed. Keep it and use a new version.')
        print(f'Keeping the existing family dataset: {OUTPUT}')
        return

    prompt_settings = json.loads(PROMPT_SOURCE.read_text())
    marker = '\0EMAIL\0'
    messages = [{'role': 'system', 'content': prompt_settings['system prompt']}, {'role': 'user', 'content': prompt_settings['classification template'].format(email=marker)}]
    signatures = []
    for folder in FOLDERS:
        path = ROOT / 'model' / folder
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        shell = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if '<think>' in shell:
            raise ValueError('This family preparation expects a template without a thinking block.')
        # Qwen2.5 does not open a thinking block; no artificial closing tag is inserted.
        head, tail = (shell + 'Classification label:').split(marker)
        head_ids = tokenizer.encode(head, add_special_tokens=True)
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        labels = {label: tokenizer.encode(' ' + label, add_special_tokens=False) for label in ['injection', 'safe']}
        if any(len(ids) != 1 for ids in labels.values()):
            raise ValueError('Both complete lowercase label words must be single tokens.')
        signatures.append({'tokenizer sha256': hashlib.sha256((path / 'tokenizer.json').read_bytes()).hexdigest(), 'head ids': head_ids, 'tail ids': tail_ids, 'label ids': labels})
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError('The three sizes do not produce identical token-space prompts.')

    source = pd.read_csv(SOURCE, engine='python', keep_default_na=False)
    rows, dropped = [], []
    for pair_id, pair in source.groupby('pair_id', sort=True):
        if len(pair) != 2 or set(pair['label']) != {0, 1} or pair['split'].nunique() != 1:
            raise ValueError('Each source pair must contain one injection and one safe email in one split.')
        clean = pair.loc[pair['label'].eq(0), 'text'].iloc[0]
        injected = pair.loc[pair['label'].eq(1), 'text'].iloc[0]
        # Recover the source generator's exact insertion point, not keyword matches.
        clean_for_matching = clean.strip()
        cuts = [match.end() for match in re.finditer(r'[.!?]\s+', clean_for_matching[:1200])]
        if not cuts:
            newline = clean_for_matching.find('\n')
            cuts = [newline + 1 if 0 < newline < 1200 else 0]
        added = len(injected) - len(clean_for_matching)
        matches = [cut for cut in cuts if added > 0 and injected[:cut] + injected[cut + added:] == clean_for_matching]
        if len(matches) != 1:
            raise ValueError(f'The exact inserted passage is ambiguous for pair {pair_id}.')
        start = matches[0]
        inserted = injected[start:start + added].strip()
        end_character = start + len(injected[start:start + added].rstrip())
        tokens = {int(row['label']): tokenizer.encode(row['text'], add_special_tokens=False) for item_index, row in pair.iterrows()}
        budget = min(len(tokens[0]), len(tokens[1]), MAX_EMAIL_TOKENS)
        end = len(tokenizer.encode(injected[:end_character], add_special_tokens=False))
        retained = tokenizer.decode(tokens[1][:budget], skip_special_tokens=False)
        if end > budget or not retained.startswith(injected[:end_character]):
            dropped.append({'pair_id': int(pair_id), 'reason': 'token cap removes part of the inserted text or its exact prefix'})
            continue

        for item_index, row in pair.iterrows():
            label = int(row['label'])
            email_ids = tokens[label][:budget]
            input_ids = head_ids + email_ids + tail_ids
            rows.append({
                'example_id': f'{pair_id}:{label}',
                'pair_id': pair_id,
                'label': label,
                'condition': 'injection' if label else 'safe',
                'source split': row['split'],
                'technique': row['technique'],
                'email tokens': budget,
                'prompt tokens': len(input_ids),
                'inserted text': inserted if label else '',
                'inserted text end token': end if label else '',
                'email text': tokenizer.decode(email_ids, skip_special_tokens=True),
                'prompt text': tokenizer.decode(input_ids, skip_special_tokens=False),
                'input token ids': json.dumps(input_ids),
            })

    data = pd.DataFrame(rows)
    train_pairs = data.loc[data['source split'].eq('train'), 'pair_id'].drop_duplicates()
    # Reserve whole pairs so versions of one email cannot cross training and validation.
    validation_pairs = set(train_pairs.sample(frac=0.2, random_state=0))
    # Keep the original CSV keys; notebooks translate them to training/validation/testing.
    data['split'] = data['source split'].replace({'train': 'fit'})
    data.loc[data['pair_id'].isin(validation_pairs), 'split'] = 'development'
    if not data.groupby('pair_id')['prompt tokens'].nunique().eq(1).all():
        raise ValueError('The final token-space inputs are not length matched within every pair.')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    manifest = {
        'dataset sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'source sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'generator sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prompt source sha256': hashlib.sha256(PROMPT_SOURCE.read_bytes()).hexdigest(),
        'model folders': FOLDERS,
        'shared prompt signature': signatures[0],
        'max email tokens': MAX_EMAIL_TOKENS,
        'seed': 0,
        'examples': len(data),
        'pairs': data['pair_id'].nunique(),
        'dropped pairs': dropped,
        'split counts': data['split'].value_counts().to_dict(),
        'selection': 'all valid source pairs, no selection by model results',
        'thinking': 'Qwen templates open no thinking block; no closing tag added',
        'scope': 'exact common inputs for the Qwen2.5 size comparison; not identical windows to the Antares tokenizer',
        'fitting started': False,
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(data.groupby(['split', 'condition']).size().rename('emails').to_string())
    print(f'Dropped {len(dropped)} pairs before any fitting.')
    print(f'Saved the exact family inputs for inspection: {OUTPUT}')


if __name__ == '__main__':
    main()
