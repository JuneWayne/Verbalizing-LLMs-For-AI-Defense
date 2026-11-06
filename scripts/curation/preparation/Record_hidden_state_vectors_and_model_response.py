"""Collect FIPI hidden state vectors and base-model responses before data selection

Both prompt versions use the original user task and the same system prompt
Classifiers read the hidden state vector at the last prompt token, before generation
Responses supply the selection completion checks and the safe reference answers
"""

import argparse
import json
import time

import numpy as np
import torch

from scripts.curation.common import (
    ROOT,
    digest,
    experiment_fingerprint,
    has_witness,
    load_model,
    load_pairs,
    messages,
    settings,
    write_json,
    write_jsonl,
)


# Construct the actual model input for each safe or injection prompt
# The shared messages function combines the configured system prompt, user instruction and chosen content
# The tokenizer's chat template then adds the role markers expected by Qwen
# Left padding allows prompts of different lengths to share a batch while keeping their last token aligned
# The input limit is checked against the complete prompt, so an injection is never silently removed by truncation
# The returned token batch is moved to the GPU for the model forward pass
def render_inputs(rows, tokenizer, config):
    """Format both prompt versions and move their token batches to the GPU"""
    texts = []
    for row, condition in rows:
        prompt_messages = messages(row, condition, config)
        text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    # tokenization creates the input IDs and padding mask expected by the model
    # the chat template already inserted special markers, so no extra markers are added here
    inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")

    # never cut off an injection or the task to meet the input limit
    if inputs.input_ids.shape[1] > config["max_input_tokens"]:
        raise ValueError("A full prompt exceeds max_input_tokens; review the limit before continuing")
    return inputs.to("cuda")


# Run one batch either to collect hidden state vectors or to generate responses
# With generate set to false, the model returns a hidden state vector sequence for each inspected block
# The last prompt token is selected because the classifiers inspect the input before any response is generated
# The embedding entry and final normalized entry are omitted, preserving the original 27 block outputs for Qwen2.5-7B
# With generate set to true, greedy decoding produces response tokens and the input tokens are removed from the returned sequence
# If the batch exceeds GPU memory, it is divided into two smaller batches and combined in the same order
# This preserves every prompt while reducing how many are processed together
def record_batch(model, tokenizer, config, batch, generate=False):
    """Collect hidden state vectors or generate responses for one batch"""

    # split a batch after an out-of-memory error without dropping or shortening a prompt
    try:
        inputs = render_inputs(batch, tokenizer, config)
        with torch.inference_mode():
            if generate:
                # greedy decoding picks the highest-scoring next token until EOS or the token limit
                result = model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"], pad_token_id=tokenizer.pad_token_id)
                return result[:, inputs.input_ids.shape[1]:].cpu().tolist()

            # a forward pass reads the full input and exposes each layer's hidden state vectors
            result = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)

        # index 0 is the embedding output and the final entry includes final normalization
        # keep the last prompt token from the remaining block outputs
        layer_vectors = []
        for hidden_state in result.hidden_states[1:-1]:
            layer_vectors.append(hidden_state[:, -1].float().cpu())

        # stacking adds the layer axis, giving [prompts, inspected layers, hidden state size]
        return torch.stack(layer_vectors, dim=1).numpy().astype(np.float16)
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise

    # leave the exception scope first so its traceback releases the failed tensors
    if "inputs" in locals():
        del inputs
    if "result" in locals():
        del result
    torch.cuda.empty_cache()

    # process the smaller halves in order, then concatenate them to preserve prompt alignment
    middle = len(batch) // 2
    print(f"Retrying batch of {len(batch)} as two smaller batches", flush=True)
    first = record_batch(model, tokenizer, config, batch[:middle], generate)
    second = record_batch(model, tokenizer, config, batch[middle:], generate)
    return first + second if generate else np.concatenate([first, second])


# Collect classifier inputs for one dataset split
# Each source pair contributes a safe prompt followed by its injection counterpart
# The batch result uses [prompt, layer, hidden state vector value], alongside matching labels and pair IDs
# These IDs let later scoring join both prompts into a pairwise selection score
# Already completed batches are skipped so an interrupted collection can continue
# Each batch is written completely before replacing its destination record
def record_states(model, tokenizer, config, output, split, limit=None):
    """Record last-prompt-token hidden state vectors with their labels and pair IDs"""
    rows = load_pairs(split)

    # each pair contributes two prompts, with safe followed by injection
    records = [(row, condition) for row in rows for condition in ("safe", "injection")]
    if limit:
        records = records[:limit * 2]
    folder = output / "states" / split
    folder.mkdir(parents=True, exist_ok=True)
    count = config["batch_size"]
    for start in range(0, len(records), count):
        path = folder / f"{start:06d}.npz"

        # completed batches remain intact when resuming an interrupted recording
        if path.exists():
            continue

        # exclude the last returned hidden state vector because it already has final normalization
        # saved shape is prompts x inspected layers x hidden state size
        batch = records[start:start + count]
        states = record_batch(model, tokenizer, config, batch)

        # keep the vectors beside the labels and IDs needed to reconstruct each safe/injection pair
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez(stream, states=states, pair_ids=[row["pair_id"] for row, _ in batch], labels=[int(condition == "injection") for _, condition in batch])
        path.with_suffix(".tmp").replace(path)
        del states
        if start % (count * 10) == 0:
            print(f"{split}: states {min(start + count, len(records))}/{len(records)}", flush=True)


# Generate the base model responses that support data selection and reference-response training
# Every pair receives a response under its safe prompt and another under its injection prompt
# The first end token determines whether the response finished naturally before the token limit
# Token IDs are kept through that end token so the safe response can later serve as a complete training target
# The witness_present field records whether the generated text contains the answer requested by the injection
# Both completion flags later determine pair eligibility, while the answer check supports model-failure selection
def generate_responses(model, tokenizer, config, output, split="candidates", limit=None):
    """Generate both responses and check completion and requested injected answers"""
    rows = load_pairs(split)

    # each pair contributes two prompts, with safe followed by injection
    records = [(row, condition) for row in rows for condition in ("safe", "injection")]
    if limit:
        records = records[:limit * 2]
    folder = output / "responses" / split
    folder.mkdir(parents=True, exist_ok=True)
    count = config.get("generation_batch_size", config["batch_size"])
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    started = time.monotonic()
    for start in range(0, len(records), count):
        path = folder / f"{start:06d}.jsonl"
        if path.exists():
            continue
        batch = records[start:start + count]
        generated = record_batch(model, tokenizer, config, batch, generate=True)
        saved = []
        for (row, condition), ids in zip(batch, generated):
            # include the first end token in the reference response and discard later batch padding
            end = next((i for i, token in enumerate(ids) if token in eos_ids), None)
            response_ids = ids if end is None else ids[:end + 1]

            # decode the response for inspection while retaining token IDs for reference training
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            saved.append({"pair_id": row["pair_id"], "condition": condition, "response": text, "response_ids": response_ids, "finished": end is not None, "generated_tokens": len(response_ids), "witness_present": has_witness(text, row["witness"])})

        # finish this batch before reporting progress, allowing later runs to skip completed batches
        write_jsonl(path, saved)
        write_json(output / f"{split}_response_progress.json", {"completed_prompts": min(start + count, len(records)), "total_prompts": len(records), "updated_unix": time.time()})
        print(f"{split}: responses {min(start + count, len(records))}/{len(records)}, elapsed {time.monotonic() - started:.0f}s", flush=True)


# Coordinate hidden state collection and response generation for the requested stage
# The model configuration determines how many block outputs and hidden state vector values are expected
# Both existing lenses are checked against those dimensions before collection starts
# The recording description identifies the inspected token position, precision and generation method
# Probe stages collect training and validation vectors, while the candidate stage collects vectors for data selection
# The final normalization weights and injection/safe unembedding rows are retained for CPU lens scoring
# The response stages generate candidate or LoRA validation answers using the same prompt setup
def run_recording(stage="all", limit=None):
    """Collect the requested split and preserve the model weights used for label scoring"""
    config, output = settings()
    fingerprint = experiment_fingerprint()
    model_settings = json.loads((ROOT / config["model"] / "config.json").read_text())

    # Qwen2.5-7B contributes 27 block outputs here, with indices 0 through 26
    layers = list(range(model_settings["num_hidden_layers"] - 1))

    # a fitted lens must have the same layer indices and vector size as the inspected model
    for item in config["lenses"].values():
        lens = torch.load(ROOT / item["path"], map_location="cpu", weights_only=True, mmap=True)
        if sorted(lens["J"]) != layers or any(tuple(matrix.shape) != (model_settings["hidden_size"],) * 2 for matrix in lens["J"].values()):
            raise ValueError("Lens layers or hidden state size differ from the configured model")
        del lens
    metadata = {"fingerprint": fingerprint, "model_config_sha256": digest(ROOT / config["model"] / "config.json"), "record_code_sha256": digest(__file__), "token_position": "last prompt token, before any response", "layers": layers, "dtype": "bfloat16 model, float16 saved states", "generation": "greedy; no inserted reasoning or classification label"}
    path = output / "recording.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("Saved recording settings differ; use a new experiment folder")
    write_json(path, metadata)
    torch.manual_seed(config["seed"])
    model, tokenizer = load_model(config)

    # the requested stage determines whether to collect probe or candidate vectors
    if stage in ("all", "states", "probe_states", "candidate_states"):
        if stage == "probe_states":
            splits = ("selector_training", "validation")
        elif stage == "candidate_states":
            splits = ("candidates",)
        else:
            splits = ("selector_training", "validation", "candidates")
        for split in splits:
            record_states(model, tokenizer, config, output, split, limit)

        # keep final RMS weights and the two label rows so scoring can run on the CPU
        # tokenize the complete leading-space label forms, each of which must occupy one token
        label_ids = [tokenizer.encode(text, add_special_tokens=False) for text in (" injection", " safe")]
        if any(len(ids) != 1 for ids in label_ids):
            raise ValueError("Both complete labels must be single tokens")

        # preserve this row order because lens_differences subtracts safe from injection
        readout = {
            "norm": model.model.norm.weight.detach().float().cpu(),
            "eps": model.model.norm.variance_epsilon,
            "unembedding": model.get_output_embeddings().weight[[ids[0] for ids in label_ids]].detach().float().cpu(),
            "label_ids": label_ids,
        }
        torch.save(readout, output / "readout.pt")
    if stage in ("all", "responses"):
        generate_responses(model, tokenizer, config, output, limit=limit)
    if stage == "validation_responses":
        generate_responses(model, tokenizer, config, output, split="validation", limit=config["lora_validation_pairs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "states", "responses", "probe_states", "candidate_states", "validation_responses"], default="all")
    parser.add_argument("--limit", type=int, help="pilot pairs per split; use a separate output folder for a pilot")
    args = parser.parse_args()
    run_recording(args.stage, args.limit)
