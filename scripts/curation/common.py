"""Share the FIPI configuration, prompt format and record-reading functions

The safe and injection versions keep the same user task and system prompt
Their content is the only part switched when constructing the two model inputs
"""

from pathlib import Path
import hashlib
import json
import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("FIPI_CONFIG", ROOT / "scripts/curation/configs/config_qwen7b.json"))

# Keep one ordered list of selection methods when preparing the eight training groups
METHODS = ["random", "text_diversity", "model_failures", "logistic_regression", "linear_svm", "neural_network", "jlens_llmail", "jlens_synthetic"]


# Calculate a content hash that identifies the bytes in one input file
# A hash changes if the file contents change, even when its name stays the same
# Experiment records use this to link a result with the configuration, prompts or model files that produced it
def digest(path):
    """Calculate a file hash to check whether its contents changed"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Read the configuration chosen for this process and locate its experiment results
# The FIPI_CONFIG environment variable selects the configuration before this module is imported
# The default is the current Qwen2.5-7B selection configuration, with FIPI_CONFIG available for an explicit override
# The function returns the settings dictionary and creates its result directory when needed
def settings():
    """Read the selected configuration and its experiment directory"""
    config = json.loads(CONFIG_PATH.read_text())
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    return config, output


# Read a dataset or result file containing one JSON object per line
# Each line becomes a dictionary, while blank lines are skipped
# Reading by file lines preserves unusual Unicode characters that can occur inside prompt strings
def read_jsonl(path):
    """Read one JSON record per line, keeping prompt text intact"""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


# Write a complete configuration or result dictionary before replacing the destination file
# The temporary file is in the same directory as its destination
# Replacing the file after writing prevents an interrupted process from leaving half of a JSON record
def write_json(path, value):
    """Replace a JSON file only after its full contents are written"""

    # replace the finished file so an interrupted write cannot leave half a record
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


# Write a sequence of prompt or response dictionaries as separate JSON lines
# Each dictionary becomes one line so later stages can read individual batches
# The completed temporary file replaces the destination only after all rows have been written
def write_jsonl(path, rows):
    """Write complete JSON records before replacing the destination file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


# Check whether generated text contains FIPI's requested injected answer
# The witness parameter is the answer requested by the injection, such as blue for an ocean-color question
# Letter case and repeated spaces are normalized before matching
# Word boundaries prevent a short answer such as blue from matching inside a longer word such as blueberry
# The result records an answer match for selection and evaluation
def has_witness(text, witness):
    """Check whether a response contains the answer requested by the injection"""

    # whole terms avoid counting blue inside blueberry, while allowing punctuation
    # normalize letter case and repeated spaces in both the response and requested answer
    text = " ".join(text.casefold().split())
    witness = " ".join(witness.casefold().split())
    return bool(witness and re.search(r"(?<!\w)" + re.escape(witness) + r"(?!\w)", text))


# Build the same user task under the safe and injection conditions
# The condition parameter chooses clean_data for safe or injected_data for injection
# The user instruction and configured system prompt remain the same in both versions
# The content tags separate the external text from the instruction in the model input
# This preserves the original experiment's prompt format across recording, selection and training
def messages(row, condition, config):
    """Build the same task prompt using either safe content or injected content"""

    # keep the original user instruction and replace only the safe or injection content
    content = row["clean_data"] if condition == "safe" else row["injected_data"]

    # the system role and user task stay fixed while the external content changes between conditions
    return [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": row["user_instruction"] + "\n\n<supplied_content>\n" + content + "\n</supplied_content>"},
    ]


# Load the local Qwen model and tokenizer used to generate the base experiment records
# The model uses BF16 precision and the configured attention implementation on the GPU
# BF16 stores each number in 16 bits while allowing a wide numerical range
# The tokenizer uses left padding so the last position of every batch item is its final prompt token
# The model is placed in evaluation mode, while the recording code disables gradient tracking for collection
def load_model(config):
    """Load the BF16 base model and its tokenizer for prompt inspection"""

    # left padding leaves the final prompt token at the last position for every batch item
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True, padding_side="left")

    # reuse the end token for left padding and rely on the attention mask to ignore padded positions
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ROOT / config["model"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    return model, tokenizer


# Read one dataset split and check that its prompt contents match the prepared experiment
# The split argument identifies probe training, validation, candidates or testing
# The manifest contains the content hash recorded when each group was prepared
# A changed prompt file is rejected so later results cannot silently use different training or testing data
def load_pairs(split):
    """Read a split after checking that its prompt records are unchanged"""
    _, output = settings()
    path = output / "splits" / f"{split}.jsonl"
    manifest = json.loads((output / "manifest.json").read_text())
    if digest(path) != manifest["split_sha256"][split]:
        raise ValueError(f"Saved {split} prompts changed after the split was fixed")
    return read_jsonl(path)


# Check that the current configuration still describes the prepared experiment
# The manifest records the original configuration hash and a shared experiment identifier
# A changed configuration requires a separate experiment directory instead of mixing records from different setups
# The returned identifier links classifier fits, candidate scores and selected training groups
def experiment_fingerprint():
    """Check that the configuration matches the prepared experiment"""
    config, output = settings()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["config_sha256"] != digest(CONFIG_PATH):
        raise ValueError("The configuration changed: use a new experiment output folder")
    return manifest["fingerprint"]
