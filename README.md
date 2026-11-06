# Verbalizing LLMs for AI Defense

In this research, I examine & experiment with 8 different ways of selecting training data from FIPI, to find which selected data helps Qwen2.5-7B-Instruct defend against prompt injections after LoRA fine-tuning.

Each method selects 256 pairs of prompts, meaning one safe prompt and its injection counterpart, then I fine-tune the model to produce the safe prompt's response when it receives the injection prompt.

## Setup

To open the notebooks, use Python 3.11 or 3.12 and run these commands from the main project folder.

```bash
uv sync --frozen --group notebook --group curation
uv run --group notebook --group curation jupyter lab
```

## Notebooks

For the results, start with the defense evaluation notebook, otherwise follow the order below to see how the data is selected and how the model is trained.

| Notebook | Purpose |
| --- | --- |
| [Workflow](notebooks/reproduce_workflow.ipynb) | See the settings and commands for each stage |
| [Inspect FIPI](notebooks/data/inspect_fipi.ipynb) | Read examples of safe and injection prompt pairs |
| [Fit Jlens](notebooks/curation/fit_jlens.ipynb) | See how both Jlens versions were fitted |
| [Collect responses](notebooks/curation/collect_fipi_responses.ipynb) | Collect responses and hidden state vectors |
| [Random selection](notebooks/curation/select_random.ipynb) | Select pairs randomly |
| [Text diversity](notebooks/curation/select_text_diversity.ipynb) | Select varied wording using TF-IDF and cosine similarity |
| [Model failures](notebooks/curation/select_model_failures.ipynb) | Select pairs where the model followed the injection's answer request |
| [Logistic Regression](notebooks/curation/select_logistic_regression.ipynb) | See how this classifier scores and selects pairs |
| [Linear SVM](notebooks/curation/select_linear_svm.ipynb) | See how this classifier scores and selects pairs |
| [2 layer neural network](notebooks/curation/select_neural_network.ipynb) | See how this classifier scores and selects pairs |
| [Jlens verbalization](notebooks/curation/select_jlens.ipynb) | Compare the pairs selected by both fitted lenses |
| [Compare selections](notebooks/curation/compare_selections.ipynb) | Compare the selected training groups |
| [LoRA training](notebooks/curation/train_lora.ipynb) | Understand the safe reference response, loss and training settings |
| [Defense evaluation](notebooks/evaluation/evaluate_defenses.ipynb) | Recreate the attack success rate tables and charts |

The notebooks explain the code and show completed results, while the scripts run response generation, fitting and training, so running the notebooks does not start training.
The Jlens selection notebook includes both fitted lenses, which is why there are 8 selection methods across 7 selection notebooks.

## Required downloads

The evaluation notebook uses the included result counts, so its charts need no model weights, dataset downloads or GPU, whereas the selection notebooks need the completed experiment records or a new run.

To repeat response generation and training, download the following data and model weights separately.

| Input | Location |
| --- | --- |
| [FIPI dataset](https://github.com/zjuynzhong/ynzhong_Rennervate_NDSS26) | `data/FIPI/raw/train_100000.json` and `test_10000.json` |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `model/Qwen2.5-7B-Instruct/` |

I also include both trained Jlens versions for Qwen2.5-7B-Instruct in the [weights release package](https://github.com/JuneWayne/Verbalizing-LLMs-For-AI-Defense/releases/download/jlens-qwen25-7b-v1/jlens-qwen25-7b-weights.zip), one fitted on LLMail prompts and another on synthetic office prompts.
Extract the package into the main project folder, using the paths in [lens_settings.json](scripts/reproduction/lens_settings.json), while the base model weights, LoRA adapters and recorded responses remain separate from this release.

I will not be open-sourcing the synthetic office prompt dataset due to privacy concerns, although the released Jlens can still be used on other prompts. Meaning, the fitted lens is provided, but retraining it on the same synthetic prompts or repeating the synthetic dataset benchmark requires the private data.

## Repeat the experiment

The workflow notebook prints the commands for preparing data, fitting probes, scoring prompts, selecting pairs and training with LoRA, using [config_qwen7b.json](scripts/curation/configs/config_qwen7b.json) for selection and [config_reference_groups.json](scripts/curation/configs/config_reference_groups.json) for training.
Before a new training run, the data and exact settings need review and approval, along with updating the file paths for the computer or cluster used.

The code is under `scripts/curation/`, with separate folders for `preparation/`, `selection/`, `training/` and `evaluation/`, while testing on external benchmarks also requires their original repositories and data.
