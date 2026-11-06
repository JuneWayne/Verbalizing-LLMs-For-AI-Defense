"""Create a fixed LLMail test CSV before fitting or evaluating any probes."""

from pathlib import Path
import hashlib
import json
import unicodedata

from transformers import AutoTokenizer
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'data/llmail/curated/probe_transfer_antares_1b.csv'
SEED = 0
MAX_TOKENS = 512
INJECTION_COUNT = 1000


def normalized_text(text):
    """Normalize text only for overlap checks; preserve the original email for classification."""
    # ignore capitalization and whitespace when checking for reused emails
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def main():
    """Select complete LLMail emails after removing overlap with training data."""
    if OUTPUT.exists():
        print(f'Keeping the existing fixed test set: {OUTPUT}')
        print(pd.read_csv(OUTPUT, engine='python')['condition'].value_counts().to_string())
        return

    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model/antares-1b', local_files_only=True)
    cami = pd.read_csv(ROOT / 'data/cami_real_pairs.csv')
    lens_fit = pd.read_csv(ROOT / 'data/llmail/curated/jlens_fit_100_balanced_under_128.csv')
    excluded = {normalized_text(text) for text in pd.concat([cami['text'], lens_fit['text']])}

    submissions = json.loads((ROOT / 'data/llmail/labelled_unique_submissions_phase2.json').read_text())
    safe_emails = json.loads((ROOT / 'data/llmail/emails_for_fp_tests.json').read_text())
    safe_keys = {normalized_text(text) for text in safe_emails}
    injection_keys = {normalized_text(text) for text, annotation in submissions.items() if str(annotation.get('attack_attempt')).lower() == 'true'}
    conflicts = safe_keys & injection_keys

    # failed submissions are not safe examples; safe emails come from the false-positive set
    rows = []
    for text, annotation in submissions.items():
        if str(annotation.get('attack_attempt')).lower() == 'true':
            rows.append({
                'text': text,
                'label': 1,
                'condition': 'injection',
                'source': 'labelled_unique_submissions_phase2.json',
                'annotation reason': annotation.get('reason', ''),
            })
    for text in safe_emails:
        rows.append({
            'text': text,
            'label': 0,
            'condition': 'safe',
            'source': 'emails_for_fp_tests.json',
            'annotation reason': 'provided safe email',
        })
    emails = pd.DataFrame(rows)
    emails['normalized text'] = emails['text'].map(normalized_text)
    overlaps = emails['normalized text'].isin(excluded)
    conflicting = emails['normalized text'].isin(conflicts)
    clean = emails.loc[~overlaps & ~conflicting].drop_duplicates('normalized text').copy()

    # keep complete emails within the CAMI token cap, so no instruction is cut off
    clean['email tokens'] = clean['text'].map(lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    eligible = clean.loc[clean['email tokens'].between(1, MAX_TOKENS)]
    injection = eligible.loc[eligible['label'].eq(1)].sample(n=min(INJECTION_COUNT, int(eligible['label'].eq(1).sum())), random_state=SEED)
    safe = eligible.loc[eligible['label'].eq(0)]
    if injection.empty or safe.empty:
        raise ValueError('Both injection and safe examples are required for the transfer test.')

    # there are no matched pairs in LLMail; the two sources are evaluated independently
    selected = pd.concat([injection, safe]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    selected.insert(0, 'email_id', selected['normalized text'].map(lambda text: hashlib.sha256(text.encode()).hexdigest()))
    selected = selected.drop(columns='normalized text')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT, index=False)
    manifest = {
        'seed': SEED,
        'max email tokens': MAX_TOKENS,
        'truncation': False,
        'injection emails': len(injection),
        'safe emails': len(safe),
        'excluded training overlaps': int(overlaps.sum()),
        'excluded conflicting rows': int(conflicting.sum()),
        'eligible counts': eligible['condition'].value_counts().to_dict(),
        'excluded over token cap': int(clean['email tokens'].gt(MAX_TOKENS).sum()),
        'excluded datasets': ['data/cami_real_pairs.csv', 'data/llmail/curated/jlens_fit_100_balanced_under_128.csv'],
        'deduplication': 'NFKC, casefold, whitespace normalization; near-duplicates may remain',
        'scope': 'Phase 2 labelled injections and unused supplied safe emails, complete emails up to 512 Antares tokens; not the whole LLMail challenge',
        'dataset sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'documentation': 'https://huggingface.co/datasets/microsoft/llmail-inject-challenge/blob/main/README.md',
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(selected['condition'].value_counts().to_string())
    print(f'Saved the fixed test set to {OUTPUT}')


if __name__ == '__main__':
    main()
