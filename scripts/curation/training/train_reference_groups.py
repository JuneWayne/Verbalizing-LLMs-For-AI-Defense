"""Train one reference-response LoRA adapter for each of the eight selected FIPI groups

The injection prompt is the input and its paired safe response is the target answer
Validation loss chooses the trained checkpoint before the fixed FIPI testing set is evaluated
"""

from pathlib import Path
import argparse
import gc
import json
import math
import time

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
import torch

from scripts.curation.common import METHODS, ROOT, digest, load_model, write_json
from scripts.curation.evaluation.evaluate_fipi import evaluate
from scripts.curation.training.run_manifest import read_config, record_manifest
from scripts.curation.training.reference_loss import task_response_loss
from scripts.curation.training.training_helpers import epoch_order, inspect_data, learning_rate


def validation_loss(model, rows):
    # average each complete answer's token loss, then average equally across pairs
    # preserve both CPU and GPU random-number sequences before measuring validation loss
    # validation must leave the next training step with the same dropout randomness
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    model.eval()
    try:
        # calculate the same reference-response loss used during training without creating gradients
        # average the complete-response losses equally across validation pairs
        with torch.no_grad():
            loss = sum(task_response_loss(model, row, next(model.parameters()).device).item() for row in rows) / len(rows)

    # restore the random-number sequences even if validation raises an error
    finally:
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
    if not math.isfinite(loss):
        raise ValueError('Non-finite validation loss')
    return loss


def best_checkpoint(history):
    # compare trained epochs by validation loss and choose the earlier epoch when losses tie
    # the original model is a baseline, not one of the trained checkpoint candidates
    trained = [row for row in history if row['epoch'] > 0]

    # choose the smallest validation loss, then the earliest epoch when losses are equal
    return min(trained, key=lambda row: (row['loss'], row['epoch']))


def train(config_path, config, method, training, validation, testing):

    output = ROOT / config['output']
    folder = output / 'adapters' / method

    # each selection method gets its own adapter directory
    # refusing an existing directory prevents replacing a completed or partially trained adapter
    folder.mkdir(parents=True, exist_ok=False)
    run = {'status': 'running', 'method': method, 'config_sha256': digest(config_path), 'started_unix': time.time()}
    write_json(folder / 'training_run.json', run)
    try:
        torch.manual_seed(config['seed'])
        model, tokenizer = load_model(config)

        # record the common original model once, before fitting the first group
        if method == config['methods'][0]:
            evaluate(model, tokenizer, config, config, {'testing': testing}, output, 'base')

        # rank and alpha control the size and scaling of the added LoRA matrices
        # only the configured projection modules in the chosen layers receive these adapters
        adapter = LoraConfig(r=config['rank'], lora_alpha=config['alpha'], lora_dropout=config['dropout'], target_modules=config['target_modules'], layers_to_transform=config['layers'], bias='none', task_type='CAUSAL_LM')
        torch.manual_seed(config['seed'])

        # attach the LoRA matrices while preserving the base model parameters
        # gradient checkpointing reduces memory by recomputing intermediate activations during backward
        model = get_peft_model(model, adapter)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        weights = [weight for weight in model.parameters() if weight.requires_grad]

        # only the LoRA matrices receive updates because the base model weights remain frozen
        optimizer = torch.optim.AdamW(weights, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        history = []

        # record the starting validation loss before any optimizer updates
        # each later epoch is compared using this same set of validation pairs
        validation_history = [{'epoch': 0, 'update': 0, 'loss': validation_loss(model, validation)}]
        write_json(folder / 'validation_loss_history.json', validation_history)
        update = 0
        for epoch in range(config['epochs']):
            # shuffle the selected pairs with the same seed rule for every adapter
            # one pass through this order is one epoch
            for index in epoch_order(config, epoch):
                update += 1

                # enable training behavior, then clear gradients left by the preceding update
                # the learning rate follows the shared schedule across all epochs
                model.train()
                optimizer.zero_grad(set_to_none=True)
                rate = learning_rate(update, config)
                for group in optimizer.param_groups:
                    group['lr'] = rate

                # train only on the paired reference answer, including its end token
                loss = task_response_loss(model, training[index], next(model.parameters()).device)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')

                # differentiate the reference-token loss with respect to the trainable LoRA matrices
                loss.backward()

                # clipping limits the total gradient size before the optimizer changes the adapter
                norm = torch.nn.utils.clip_grad_norm_(weights, config['gradient_clip'])
                if not torch.isfinite(norm):
                    raise ValueError('Non-finite gradients')

                # apply the clipped gradients to the LoRA matrices to complete this update
                optimizer.step()
                history.append({'epoch': epoch + 1, 'update': update, 'pair_id': training[index]['pair_id'], 'loss': loss.item(), 'learning_rate': rate})
                del loss
                write_json(folder / 'loss_history.json', history)
                write_json(folder / 'progress.json', {'stage': 'training', 'update': update, 'total_updates': config['total_updates']})
                print(f'{method}: update {update}/{config["total_updates"]}, loss {history[-1]["loss"]:.4f}', flush=True)

            # retain an epoch checkpoint before measuring its validation loss
            # keeping every epoch makes the final checkpoint choice independent of testing results
            checkpoint = folder / f'epoch_{epoch + 1}'
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            validation_history.append({'epoch': epoch + 1, 'update': update, 'loss': validation_loss(model, validation)})

            # validation uses the same reference-answer loss without updating the adapter
            write_json(folder / 'validation_loss_history.json', validation_history)

        # select by validation loss, then reload that epoch's adapter weights
        # the last trained epoch and the selected epoch can therefore differ
        selected = best_checkpoint(validation_history)
        write_json(folder / 'selected_checkpoint.json', {**selected, 'path': f'epoch_{selected["epoch"]}', 'selection_rule': 'lowest validation loss, earlier epoch breaks ties'})
        del optimizer, weights
        gc.collect()
        torch.cuda.empty_cache()
        model.gradient_checkpointing_disable()
        set_peft_model_state_dict(model, load_peft_weights(folder / f'epoch_{selected["epoch"]}', device='cpu'))
        model.eval()

        # testing never changes which checkpoint is chosen
        evaluate(model, tokenizer, config, config, {'testing': testing}, output, method)
        run.update(status='complete', updates=update, selected_epoch=selected['epoch'], finished_unix=time.time())
    except BaseException as error:
        run.update(status='failed', error=repr(error), finished_unix=time.time())
        raise
    finally:
        write_json(folder / 'training_run.json', run)


# Run the same Python entry point on a workstation, cloud GPU or cluster
# The check-only option validates inputs and returns before any GPU model is loaded
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'scripts/curation/configs/config_reference_groups.json')
    method_choice = parser.add_mutually_exclusive_group()
    method_choice.add_argument('--method', choices=METHODS)
    method_choice.add_argument('--method-index', type=int, choices=range(8))
    parser.add_argument('--check-only', action='store_true', help='Validate the prepared data and settings without training or generating responses')
    args = parser.parse_args()
    if not args.check_only and args.method is None and args.method_index is None:
        parser.error('Choose --method or --method-index, or use --check-only')
    path = args.config.resolve()
    config = read_config(path)

    # All source, token-length and split checks run before a model can be trained
    # The check-only command also works on a computer without a GPU
    groups, validation, testing = inspect_data(config)
    if args.check_only:
        print(json.dumps({'methods': list(groups), 'training_pairs_per_method': config['training_pairs'], 'validation_pairs': len(validation), 'testing_pairs': len(testing), 'training_started': False}, indent=2))
        raise SystemExit(0)
    method = args.method
    if method is None:
        if args.method_index >= len(config['methods']):
            parser.error('The method index is outside this configuration')
        method = config['methods'][args.method_index]
    if method not in groups:
        parser.error('The selected method is absent from this configuration')
    torch.set_num_threads(config.get('cpus', 4))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training needs a CUDA GPU with BF16 support')

    # Create the input record automatically, then train only the requested data group
    record_manifest(path, config)
    train(path, config, method, groups[method], validation, testing)


if __name__ == '__main__':
    main()
