# Verbalizing LLMs for AI Defense

In this research, I examine & experiment with 8 ways of selecting training data from FIPI, then fine-tune Qwen2.5-7B-Instruct on 256 pairs selected by each method, to find which data helps the model defend against prompt injections.

## Repository structure

```text
notebooks/                 Experiment explanations and executed results
scripts/curation/
    preparation/           Prepare prompts, responses and hidden state vectors
    selection/             Fit classifiers and select training pairs
    training/              Train LoRA adapters with safe reference responses
    evaluation/            Generate and score benchmark responses
    configs/               Data-selection and training settings
    tests/                 Check the calculations and data handling
```

## Environment setup

Use Python 3.11 or 3.12, then install the dependencies and open Jupyter from the main project folder:

```bash
python -m pip install uv
uv sync --frozen --group notebook --group curation
uv run --group notebook --group curation jupyter lab
```

Response generation and LoRA training use a CUDA GPU with BF16 support, while the published evaluation charts can be recreated without model weights or a GPU.

## Data and model downloads

| Download | Location after download or extraction |
| --- | --- |
| [FIPI dataset](https://github.com/zjuynzhong/ynzhong_Rennervate_NDSS26) | `data/FIPI/raw/train_100000.json` and `data/FIPI/raw/test_10000.json` |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `model/Qwen2.5-7B-Instruct/` |
| [Both trained Jlens weights](https://github.com/JuneWayne/Verbalizing-LLMs-For-AI-Defense/releases/download/jlens-qwen25-7b-v1/jlens-qwen25-7b-weights.zip) | Extract into the main project folder |

I provide one Jlens fitted on LLMail prompts and another fitted on synthetic office prompts, with their fitting settings in [lens_settings.json](scripts/reproduction/lens_settings.json).

I will not be open-sourcing the synthetic office prompt dataset due to privacy concerns, although the released lens can still be used on other prompts, while retraining it on the original data or repeating the synthetic benchmark requires that private dataset.

## Running experiments

### Data selection

Start with [Inspect FIPI](notebooks/data/inspect_fipi.ipynb), then the [workflow notebook](notebooks/reproduce_workflow.ipynb) explains the settings and commands for each stage.
To repeat response generation, probe fitting and selection:

```bash
FIPI_SKIP_NOTEBOOKS=1 uv run --group curation python -m scripts.curation.preparation.preparation --stage all
```

The configuration is [config_qwen7b.json](scripts/curation/configs/config_qwen7b.json), and each method selects 256 pairs from the shared candidate pool.
The selection notebooks show the completed run, so repeating them on a new computer requires those experiment records or a new selection run.

### LoRA fine-tuning

After selection, the workflow notebook prints the commands to export the reference responses and prepare validation data, then the following commands check the prepared inputs and train the random-selection adapter:

```bash
uv run --group curation python -m scripts.curation.training.train_reference_groups --check-only
uv run --group curation python -m scripts.curation.training.train_reference_groups --method random
```

Use [config_reference_groups.json](scripts/curation/configs/config_reference_groups.json) for the shared training settings and replace `random` with another selection method to train that group.
The first command does not train, while the second trains with the injection prompt as input and its safe counterpart's generated response as the reference answer.
A `manifest.json` file automatically records the settings and file hashes, so changes to the inputs can be detected.

### Evaluation

The trainer evaluates its chosen checkpoint on FIPI after training, while [Defense evaluation](notebooks/evaluation/evaluate_defenses.ipynb) recreates the published attack success rate tables and charts from the included result counts.
External benchmark evaluators are under `scripts/curation/evaluation/` and also require their original repositories and data.

## Notebooks

<details>
<summary>Notebook order and purpose</summary>

| Notebook | Purpose |
| --- | --- |
| [Workflow](notebooks/reproduce_workflow.ipynb) | Settings and commands for each stage |
| [Inspect FIPI](notebooks/data/inspect_fipi.ipynb) | Read safe and injection prompt pairs |
| [Fit Jlens](notebooks/curation/fit_jlens.ipynb) | Compare both lens-fitting methods |
| [Collect responses](notebooks/curation/collect_fipi_responses.ipynb) | Inspect responses and hidden state vectors |
| [Random selection](notebooks/curation/select_random.ipynb) | Select pairs randomly |
| [Text diversity](notebooks/curation/select_text_diversity.ipynb) | Select varied wording |
| [Model failures](notebooks/curation/select_model_failures.ipynb) | Select pairs using observed injection failures |
| [Logistic Regression](notebooks/curation/select_logistic_regression.ipynb) | Inspect pairwise selection scores |
| [Linear SVM](notebooks/curation/select_linear_svm.ipynb) | Inspect pairwise selection scores |
| [2 layer neural network](notebooks/curation/select_neural_network.ipynb) | Inspect pairwise selection scores |
| [Jlens verbalization](notebooks/curation/select_jlens.ipynb) | Compare both fitted lenses as selectors |
| [Compare selections](notebooks/curation/compare_selections.ipynb) | Compare the selected training groups |
| [LoRA training](notebooks/curation/train_lora.ipynb) | Explain the reference-response loss and settings |
| [Defense evaluation](notebooks/evaluation/evaluate_defenses.ipynb) | Recreate the result tables and charts |

</details>
