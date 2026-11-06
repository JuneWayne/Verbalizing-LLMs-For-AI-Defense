"""Read completed data selection results for the notebook explanations

configuration reads the Qwen2.5-7B experiment settings
selection returns one method's selected prompt pairs
validation returns the layer accuracies used to choose that detector's layer
"""

from pathlib import Path
import hashlib
import json

import pandas as pd

from scripts.curation.common import ROOT


# Read the experiment settings used by the notebook result tables
# The selected configuration is the completed Qwen2.5-7B run
# Its content hash is compared with the experiment manifest before results are returned
# The returned config dictionary describes the model and selection settings, while output identifies the matching result directory
def configuration():
    """Read the 7B configuration and check that it matches the recorded experiment"""
    path = ROOT / "scripts/curation/configs/config_qwen7b.json"
    config = json.loads(path.read_text())
    output = ROOT / config["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    # check that the notebook settings match the experiment that produced these results
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("The 7B configuration differs from the completed experiment")
    return config, output


# Return one method's original selected training pairs for notebook inspection
# The method argument identifies which of the eight training groups to read
# Its experiment identifier and pair-file hash must match the completed run
# The count and unique pair IDs are checked against the configured selection size
# The returned table contains the original task, both content versions and the selection scores
def selection(method):
    """Read one completed selection and check its experiment and file contents"""
    config, output = configuration()
    folder = output / "selections" / method
    manifest = json.loads((folder / "manifest.json").read_text())
    experiment = json.loads((output / "manifest.json").read_text())
    path = folder / "pairs.jsonl"
    # pair IDs, selected text and experiment settings must belong to the same run
    if manifest["method"] != method or manifest["fingerprint"] != experiment["fingerprint"]:
        raise ValueError("The selected pairs belong to another experiment")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["pairs_sha256"]:
        raise ValueError("The selected pair file changed after selection")
    # iterate over file lines because Unicode line separators can occur inside prompt strings
    with path.open() as stream:
        rows = pd.DataFrame(json.loads(line) for line in stream if line.strip())
    if len(rows) != config["selected_pairs"] or rows.pair_id.duplicated().any():
        raise ValueError("The selection has missing or duplicated pairs")
    return rows


# Read the layer accuracies used to choose a detector's hidden state vector layer
# The fitting record identifies the experiment and hidden state recording that produced those accuracies
# Those identifiers are checked before the layer table is returned
# The notebook uses this table to plot injection and safe accuracy without refitting the detector
def validation(method):
    """Read the recorded layer accuracies without loading or fitting a classifier"""
    config, output = configuration()
    fit = json.loads((output / "selectors" / f"{method}_fit.json").read_text())
    experiment = json.loads((output / "manifest.json").read_text())
    # layer accuracy must refer to the same hidden state recording as the selected pairs
    signature = fit["signature"]
    if signature["fingerprint"] != experiment["fingerprint"]:
        raise ValueError("The validation scores belong to another experiment")
    if signature["recording_sha256"] != hashlib.sha256((output / "recording.json").read_bytes()).hexdigest():
        raise ValueError("The response recording differs from the detector fitting record")
    return pd.read_csv(output / "selectors" / f"{method}_validation.csv")
