"""Inspect the imported FIPI files without changing their contents or running models."""

from collections import Counter
from pathlib import Path
from statistics import median
import csv
import hashlib
import json


root = Path(__file__).resolve().parents[2] / 'data' / 'FIPI'
report = {'splits': {}, 'overlap': {}}
split_sets = {}
samples = []
issues = []

for split, filename in [('training', 'train_100000.json'), ('testing', 'test_10000.json')]:
    path = root / 'raw' / filename
    raw_bytes = path.read_bytes()
    rows = json.loads(raw_bytes)
    checks = Counter()
    missing = Counter()
    shown_methods = set()
    word_counts = []

    for row in rows:
        text = row['injected_data']
        start = row['injected_position']['start']
        end = row['injected_position']['end']
        injection = row['attack_instruction']
        clean = row['clean_data']
        instruction = row['user_instruction']
        witness = row['witness']
        word_counts.append(len(text.split()))
        missing.update(key for key, value in row.items() if value is None or value == '')

        # FIPI includes the final character in its end index; Python slices do not.
        checks['valid_character_bounds'] += 0 <= start <= end < len(text)
        checks['exact_injection_span'] += text[start:end + 1] == injection
        checks['nonempty_injection'] += bool(injection.strip())

        # Insertion can add a space on either side. Do not strip or rewrite the clean prompt.
        restored_versions = [text[:start - left] + text[end + 1 + right:] for left in (0, 1) for right in (0, 1) if start >= left]
        checks['clean_content_reconstructed'] += clean in restored_versions

        # The witness is the expected answer to the injected request, not an injection/safe label.
        checks['witness_in_clean_content'] += bool(witness) and witness.casefold() in clean.casefold()
        checks['witness_in_task_instruction'] += bool(witness) and witness.casefold() in instruction.casefold()
        if not instruction.strip():
            issues.append({'split': split, 'data_id': row['data_id'], 'issue': 'empty_task_instruction'})
        if witness and witness.casefold() in instruction.casefold():
            issues.append({'split': split, 'data_id': row['data_id'], 'issue': 'witness_in_task_instruction'})
        if text[start:end + 1] != injection or clean not in restored_versions:
            issues.append({'split': split, 'data_id': row['data_id'], 'issue': 'injection_span_or_clean_content_mismatch'})

        # Export one complete example per injection method and split, with its clean counterpart.
        if row['attack_method'] not in shown_methods:
            shown_methods.add(row['attack_method'])
            for condition, content in [('safe', clean), ('injection', text)]:
                samples.append({'split': split, 'data_id': row['data_id'], 'user_prompt_id': row['user_prompt_id'], 'condition': condition, 'task': row['user_case']['subtask'], 'method': row['attack_method'], 'user_instruction': instruction, 'content': content, 'injection_text': injection if condition == 'injection' else '', 'witness': witness})

    split_sets[split] = {
        'record_ids': {row['data_id'] for row in rows},
        'source_prompt_ids': {row['user_prompt_id'] for row in rows},
        'task_and_clean_content': {(row['user_instruction'], row['clean_data']) for row in rows},
        'clean_content_only': {row['clean_data'] for row in rows},
        'task_and_injected_content': {(row['user_instruction'], row['injected_data']) for row in rows},
    }
    report['splits'][split] = {
        'path': str(path.relative_to(root)),
        'sha256': hashlib.sha256(raw_bytes).hexdigest(),
        'bytes': len(raw_bytes),
        'records': len(rows),
        'fields': sorted(rows[0]),
        'schema_counts': dict(Counter(', '.join(sorted(row)) for row in rows)),
        'missing_or_empty_fields': dict(missing),
        'unique_counts': {key: len(values) for key, values in split_sets[split].items()},
        'injection_methods': dict(Counter(row['attack_method'] for row in rows)),
        'task_categories': dict(Counter(row['user_case']['type'] for row in rows)),
        'unique_category_and_subtask_pairs': len({(row['user_case']['type'], row['user_case']['subtask']) for row in rows}),
        'injected_content_whitespace_words': {'min': min(word_counts), 'median': median(word_counts), 'max': max(word_counts)},
        'checks': dict(checks),
    }

# Compare source groups, not just record IDs, before treating the testing split as independent.
for key in split_sets['training']:
    report['overlap'][key] = len(split_sets['training'][key] & split_sets['testing'][key])
shared_clean = split_sets['training']['clean_content_only'] & split_sets['testing']['clean_content_only']
report['overlap']['records_with_shared_clean_content'] = {}
for split in ['training', 'testing']:
    path = root / report['splits'][split]['path']
    rows = json.loads(path.read_text())
    shared_rows = [row for row in rows if row['clean_data'] in shared_clean]
    report['overlap']['records_with_shared_clean_content'][split] = len(shared_rows)
    for row in shared_rows:
        issues.append({'split': split, 'data_id': row['data_id'], 'issue': 'clean_content_shared_between_splits'})

# These are inspection outputs only. No training split, labels, or source text are changed.
output = root / 'inspection'
output.mkdir(exist_ok=True)
(output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
for filename, records, fields in [('examples.csv', samples, list(samples[0])), ('review_rows.csv', issues, ['split', 'data_id', 'issue'])]:
    with (output / filename).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

print(json.dumps(report, indent=2))
print(f'Saved {len(samples)} sample rows and {len(issues)} review flags in {output}')
