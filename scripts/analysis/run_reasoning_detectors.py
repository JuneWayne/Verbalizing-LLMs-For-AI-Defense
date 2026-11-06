"""Compare detectors after reasoning, using existing CAMI and LLMail lenses.

Run from the repository root:
python -m scripts.analysis.run_reasoning_detectors --model antares-1b
"""

from pathlib import Path
import argparse
import fcntl
import gc
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import GraniteMoeHybridRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
import jlens
import numpy as np
import pandas as pd
import torch

from scripts.analysis import reasoning_detectors as experiment
from scripts.analysis.experiment_files import file_hash, read_tensor, save_json


def prepare(model_key, directory, max_new_tokens, device, gpu_memory='9GiB', cpu_memory='8GiB', synthetic_path=None, llmail_path=None, example_ids=None):
    """Save the exact inputs, model files, and lenses before collecting responses."""
    emails, excluded_emails = experiment.load_emails(synthetic_path, llmail_path)
    if example_ids is not None:
        # Output inspection uses an explicit small subset, never a partial accuracy run.
        if len(example_ids) != len(set(example_ids)) or not set(example_ids).issubset(emails['example_id']):
            raise ValueError('Inspection IDs must be unique and present in the prepared data.')
        emails = emails.set_index('example_id').loc[example_ids].reset_index()
    lens_files = experiment.lens_inventory(model_key)
    if not lens_files['available'].all():
        raise FileNotFoundError('A registered lens file is missing. Inspect the lens inventory.')
    model_path = experiment.MODELS[model_key].model_path(experiment.ROOT)
    model_config = json.loads((model_path / 'config.json').read_text())
    # Qwen3.5 stores its language-model settings inside a multimodal configuration.
    model_config = model_config.get('text_config', model_config)
    # The trained lenses use every source block except the final model block.
    layers = list(range(model_config['num_hidden_layers'] - 1))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # All detectors and both lenses receive the same generated reasoning and label position.
    emails['prompt text'] = [experiment.build_prompt(tokenizer, text) for text in emails['email text']]
    for path in lens_files['lens path']:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if checkpoint['source_layers'] != layers or checkpoint['d_model'] != model_config['hidden_size']:
            raise ValueError(f'The lens does not match this model: {path}')
        del checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    # Hashes identify the exact files used, not just their names.
    data_files = [Path(synthetic_path) if synthetic_path else experiment.CAMI_PATH, Path(llmail_path) if llmail_path else experiment.LLMAIL_PATH, experiment.CAMI_LENS_DATA, experiment.LLMAIL_LENS_DATA]
    code_files = [Path(__file__), Path(experiment.__file__), experiment.ROOT / 'scripts/analysis/detector_methods.py', experiment.ROOT / 'scripts/analysis/experiment_files.py']
    settings = {
        'model': model_key,
        'model path': str(model_path),
        'email data': [str(path) for path in data_files[:2]],
        'max new tokens': max_new_tokens,
        'device': device,
        'layers': layers,
        'width': model_config['hidden_size'],
        'system prompt': experiment.SYSTEM_PROMPT,
        'classification prompt': experiment.CLASSIFICATION_PROMPT,
        'reasoning': 'Enabled; model generates its own closing tag when applicable',
        'state position': 'Exact generated prefix before the final label token; label excluded',
        'source sha256': {str(path): file_hash(path) for path in data_files},
        'lens sha256': {path: file_hash(Path(path)) for path in lens_files['lens path']},
        'code sha256': {str(path): file_hash(path) for path in code_files},
        'excluded lens training examples': len(excluded_emails),
        'examples': len(emails),
        'inspection example ids': example_ids,
        'model config sha256': file_hash(model_path / 'config.json'),
        'tokenizer sha256': file_hash(model_path / 'tokenizer.json'),
        'tokenizer config sha256': file_hash(model_path / 'tokenizer_config.json'),
    }
    weight_index_path = model_path / 'model.safetensors.index.json'
    weight_files = ['model.safetensors']
    if weight_index_path.exists():
        weight_files = sorted(set(json.loads(weight_index_path.read_text())['weight_map'].values()))
    settings['model shard sha256'] = {name: file_hash(model_path / name) for name in weight_files}
    settings['memory plan'] = {'GPU weights': gpu_memory, 'CPU weights': cpu_memory}
    # Settings can change before generation; collected results always keep their original settings.
    run_started = any((directory / 'chunks').glob('*.npz')) or (directory / 'evaluation.json').exists()
    save_json(directory / 'run.json', settings, fixed=run_started)
    emails.to_csv(directory / 'emails.csv', index=False)
    excluded_emails.to_csv(directory / 'excluded_lens_training_pairs.csv', index=False)
    lens_files.to_csv(directory / 'lenses.csv', index=False)
    return emails, lens_files, tokenizer, settings


def collect(data, tokenizer, settings, directory, limit=0):
    """Generate responses once and save hidden states in recoverable batches."""
    model = None
    answer_records = []
    batch_folder = directory / 'chunks'
    batch_folder.mkdir(exist_ok=True)
    run_hash = file_hash(directory / 'run.json')
    for start in range(0, len(data), 16):
        # Stop only at a batch boundary, so saved files always contain complete batches.
        if limit and start >= limit:
            break
        batch_emails = data.iloc[start:start + 16]
        batch_path = batch_folder / f'{start:05d}.npz'
        if batch_path.exists():
            # Reuse saved work only when both settings and batch contents still match.
            saved_hashes = json.loads(batch_path.with_suffix('.json').read_text())
            if saved_hashes != {'run sha256': run_hash, 'chunk sha256': file_hash(batch_path)}:
                raise ValueError('The saved generation belongs to different settings or changed files.')
            with np.load(batch_path, allow_pickle=False) as saved:
                states = saved['hidden']
                records = [json.loads(value) for value in saved['records']]
        else:
            if model is None:
                dtype = torch.bfloat16 if settings['device'].startswith('cuda') else torch.float32
                # Automatic placement allows larger models to use CPU memory when needed.
                model_loading = {'device_map': 'cpu'}
                if settings['device'].startswith('cuda'):
                    gpu_index = torch.device(settings['device']).index or 0
                    model_loading = {'device_map': 'auto', 'max_memory': {gpu_index: settings['memory plan']['GPU weights'], 'cpu': settings['memory plan']['CPU weights']}}
                model = AutoModelForCausalLM.from_pretrained(settings['model path'], dtype=dtype, local_files_only=True, **model_loading).eval().requires_grad_(False)
            batch_states = []
            records = []
            for email in batch_emails.to_dict('records'):
                email_states, record = experiment.generate_and_read(model, tokenizer, email['prompt text'], settings['layers'], settings['max new tokens'])
                record.update({column: email[column] for column in ['example_id', 'pair_id', 'dataset', 'split', 'condition', 'label']})
                batch_states.append(email_states)
                records.append(record)
            # Array shape: emails in this batch, source layers, hidden-state width.
            states = np.stack(batch_states)
            temporary_path = batch_path.with_suffix('.partial')
            with temporary_path.open('wb') as file:
                np.savez(file, hidden=states, records=np.array([json.dumps(record) for record in records]))
            temporary_path.replace(batch_path)
            save_json(batch_path.with_suffix('.json'), {'run sha256': run_hash, 'chunk sha256': file_hash(batch_path)})
        if [record['example_id'] for record in records] != batch_emails['example_id'].tolist() or states.shape != (len(batch_emails), len(settings['layers']), settings['width']):
            raise ValueError('Saved states do not match the expected emails and layers.')
        answer_records.extend(records)
        save_json(directory / 'progress.json', {'complete': len(answer_records) == len(data), 'completed examples': len(answer_records), 'expected examples': len(data)})
        print(f"{settings['model']}: {len(answer_records)}/{len(data)} responses", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if len(answer_records) == len(data):
        pd.DataFrame(answer_records).to_json(directory / 'answers.jsonl', orient='records', lines=True)
    return len(answer_records) == len(data)


def load_states(data, settings, directory):
    """Verify generation files before any training or accuracy calculation."""
    answer_records = []
    state_batches = []
    run_hash = file_hash(directory / 'run.json')
    for start in range(0, len(data), 16):
        batch_path = directory / f'chunks/{start:05d}.npz'
        saved_hashes = json.loads(batch_path.with_suffix('.json').read_text())
        if saved_hashes != {'run sha256': run_hash, 'chunk sha256': file_hash(batch_path)}:
            raise ValueError('A generation file or its settings changed.')
        with np.load(batch_path, allow_pickle=False) as saved:
            state_batches.append(saved['hidden'])
            answer_records.extend(json.loads(value) for value in saved['records'])
    answers = pd.DataFrame(answer_records)
    if answers['example_id'].tolist() != data['example_id'].tolist():
        raise ValueError('The generated responses do not match the prepared emails.')
    states = np.concatenate(state_batches)
    if states.shape != (len(data), len(settings['layers']), settings['width']):
        raise ValueError('Hidden-state dimensions do not match the selected model.')
    if not np.isfinite(states[answers['state available']]).all():
        raise ValueError('A readable response has an invalid hidden state.')
    return states, answers


def project(states, answers, inventory, tokenizer, settings):
    """Read each saved layer with Jlens and with the model's output weights alone."""

    model_path = Path(settings['model path'])
    model_config = json.loads((model_path / 'config.json').read_text())
    # The Qwen3.5 checkpoint keeps language weights under a nested model prefix.
    weight_prefix = 'model.language_model' if 'text_config' in model_config else 'model'
    model_config = model_config.get('text_config', model_config)
    normalization_classes = {'qwen2': Qwen2RMSNorm, 'llama': LlamaRMSNorm, 'granitemoehybrid': GraniteMoeHybridRMSNorm, 'qwen3_5_text': Qwen3_5RMSNorm}
    dtype = torch.bfloat16 if settings['device'].startswith('cuda') else torch.float32
    # Both lenses use the model's actual final normalization and vocabulary weights.
    final_normalization = normalization_classes[model_config['model_type']](settings['width'], eps=model_config['rms_norm_eps'])
    final_normalization.load_state_dict({'weight': read_tensor(model_path, weight_prefix + '.norm.weight')})
    final_normalization = final_normalization.to(device=settings['device'], dtype=dtype).eval().requires_grad_(False)
    output_weight_name = weight_prefix + '.embed_tokens.weight' if model_config['tie_word_embeddings'] else 'lm_head.weight'
    output_weights = read_tensor(model_path, output_weight_name).to(device=settings['device'], dtype=dtype)
    lens_variants = [('not applicable', None)] + list(inventory[['lens training', 'lens path']].itertuples(index=False, name=None))
    score_tables = []
    for lens_training, lens_path in lens_variants:
        lens = None
        if lens_path:
            saved = torch.load(lens_path, map_location='cpu', weights_only=True, mmap=True)
            lens = jlens.JacobianLens(saved['J'], n_prompts=saved['n_prompts'], d_model=saved['d_model'])
        detector = experiment.JLENS if lens else experiment.LOGIT_LENS
        for layer_position, layer in enumerate(settings['layers']):
            # Move only this layer's Jacobian to the device, keeping memory use bounded.
            if lens:
                lens.jacobians[layer] = lens.jacobians[layer].to(settings['device'])
            layer_scores = answers[['example_id', 'dataset', 'split', 'condition', 'label']].copy()
            layer_scores = layer_scores.assign(detector=detector, layer=layer, **{'lens training': lens_training, 'score': np.nan, 'injection logit': np.nan, 'safe logit': np.nan, 'top tokens': ''})
            valid_indices = np.flatnonzero(answers['state available'].to_numpy())
            for start in range(0, len(valid_indices), 32):
                indices = valid_indices[start:start + 32]
                layer_states = torch.tensor(states[indices, layer_position], device=settings['device'])
                with torch.inference_mode():
                    # Jlens estimates the final state; Logit lens uses the current state directly.
                    estimated_states = lens.transport(layer_states, layer) if lens else layer_states
                    logits = torch.nn.functional.linear(final_normalization(estimated_states.to(dtype)), output_weights).float()
                # Match the whitespace of the actual generated label, rather than guessing a token ID.
                label_token_ids = torch.tensor([[answers.iloc[email_index]['label token ids'][word][0] for word in ['injection', 'safe']] for email_index in indices], device=logits.device)
                label_logits = logits.gather(1, label_token_ids).cpu().numpy()
                layer_scores.loc[indices, 'injection logit'] = label_logits[:, 0]
                layer_scores.loc[indices, 'safe logit'] = label_logits[:, 1]
                layer_scores.loc[indices, 'score'] = label_logits[:, 0] - label_logits[:, 1]
                top_token_ids = logits.topk(5, dim=-1).indices.cpu().tolist()
                layer_scores.loc[indices, 'top tokens'] = [json.dumps([tokenizer.decode([token]) for token in tokens]) for tokens in top_token_ids]
            score_tables.append(layer_scores)
            if lens:
                lens.jacobians[layer] = lens.jacobians[layer].cpu()
        print(f'{detector}: {lens_training}', flush=True)
        del lens
    return pd.concat(score_tables, ignore_index=True)


def fit_and_evaluate(states, answers, settings, directory, lens_scores):
    """Fit probes on training, choose settings on validation, then score testing."""
    has_state = answers['state available'].to_numpy()
    is_training = answers['split'].eq('training').to_numpy() & has_state
    if set(answers.loc[is_training, 'label']) != {0, 1}:
        raise ValueError('Probe training needs readable hidden states from both classes.')
    score_tables = [lens_scores]
    checkpoint_folder = directory / 'checkpoints'
    checkpoint_folder.mkdir(exist_ok=True)
    training_labels = torch.tensor(answers.loc[is_training, 'label'].to_numpy(), dtype=torch.float32, device=settings['device'])
    for layer_position, layer in enumerate(settings['layers']):
        # All three probes see the same state vectors and training labels at this layer.
        training_states = torch.tensor(states[is_training, layer_position], device=settings['device'])
        scoring_states = torch.tensor(states[has_state, layer_position], device=settings['device'])
        for detector, kind in experiment.PROBES.items():
            checkpoint = experiment.fit_probe(training_states, training_labels, scoring_states, kind)
            checkpoint.update({'model': settings['model'], 'layer': layer, 'detector': detector, 'run sha256': file_hash(directory / 'run.json')})
            torch.save(checkpoint, checkpoint_folder / f'{layer:02d}_{kind.replace(" ", "_")}.pt')
            layer_scores = answers[['example_id', 'dataset', 'split', 'condition', 'label']].copy()
            layer_scores = layer_scores.assign(detector=detector, layer=layer, **{'lens training': 'not applicable', 'score': np.nan})
            layer_scores.loc[has_state, 'score'] = checkpoint['values'].numpy().astype(np.float64)
            score_tables.append(layer_scores)
        print(f"{settings['model']}: trained probes at layer {layer}", flush=True)
    scores = pd.concat(score_tables, ignore_index=True)
    scores.to_csv(directory / 'scores.csv', index=False)
    # Layer and threshold choices use validation only, never testing accuracy.
    validation, selected = experiment.select_detectors(scores)
    selected.to_csv(directory / 'selected_detectors.csv', index=False)
    validation.to_csv(directory / 'validation_accuracy.csv', index=False)
    expected_detectors = len(experiment.PROBES) + 1 + lens_scores.loc[lens_scores['detector'].eq(experiment.JLENS), 'lens training'].nunique()
    if len(selected) != expected_detectors:
        raise ValueError('Not every detector has valid validation scores from both classes.')
    predictions, accuracy = experiment.score_predictions(scores, selected, answers)
    predictions.to_csv(directory / 'predictions.csv', index=False)
    accuracy.to_csv(directory / 'accuracy.csv', index=False)
    answers.groupby(['dataset', 'split', 'condition', 'format status']).size().rename('emails').to_csv(directory / 'response_formats.csv')
    save_json(directory / 'evaluation.json', {'complete': True, 'reasoning enabled': True, 'run sha256': file_hash(directory / 'run.json'), 'predictions sha256': file_hash(directory / 'predictions.csv'), 'accuracy sha256': file_hash(directory / 'accuracy.csv'), 'selected detectors sha256': file_hash(directory / 'selected_detectors.csv')})
    print(accuracy.to_string(index=False), flush=True)


def main():
    """Run only the requested stage, using one writer per result folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=experiment.MODEL_KEYS, required=True)
    parser.add_argument('--stage', choices=['prepare', 'collect', 'evaluate', 'all'], default='all')
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory', default='9GiB')
    parser.add_argument('--cpu-memory', default='8GiB')
    parser.add_argument('--run-name', default=experiment.RESULT_FOLDER)
    parser.add_argument('--synthetic-data', type=Path, help='Prepared full-length email CSV; omitted uses the historical shortened data.')
    parser.add_argument('--llmail-data', type=Path, help='Prepared LLMail testing CSV; omitted uses the historical test CSV.')
    parser.add_argument('--limit', type=int, default=0, help='Collect this many responses, rounded up to 16; zero collects the full dataset.')
    arguments = parser.parse_args()
    if arguments.max_new_tokens < 1 or arguments.limit < 0:
        parser.error('Use a positive token budget and a nonnegative collection limit.')
    if not arguments.run_name or Path(arguments.run_name).name != arguments.run_name or arguments.run_name in ['.', '..']:
        parser.error('Use a single folder name for --run-name.')
    directory = experiment.ROOT / 'model/artifacts' / arguments.model / arguments.run_name
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision('highest')
    with (directory / 'writer.lock').open('a') as lock:
        # Refuse a second writer instead of risking mixed or overwritten results.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        emails, lens_files, tokenizer, settings = prepare(arguments.model, directory, arguments.max_new_tokens, arguments.device, arguments.gpu_memory, arguments.cpu_memory, arguments.synthetic_data, arguments.llmail_data)
        if arguments.stage == 'prepare':
            print(lens_files.to_string(index=False))
            print(emails.groupby(['dataset', 'split', 'condition']).size().to_string())
            return
        if arguments.stage in ['collect', 'all']:
            collection_complete = collect(emails, tokenizer, settings, directory, arguments.limit)
            if arguments.stage == 'collect' or not collection_complete:
                return
        states, answers = load_states(emails, settings, directory)
        lens_scores = project(states, answers, lens_files, tokenizer, settings)
        lens_scores.to_csv(directory / 'lens_scores.csv', index=False)
        fit_and_evaluate(states, answers, settings, directory, lens_scores)


if __name__ == '__main__':
    main()
