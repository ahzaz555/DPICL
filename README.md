# PriTabICL

This repository contains the implementation used for the experiments reported in the submission.

The code is released with minimal structural modification from the experimental version used to obtain the reported results. The main implementation is intentionally kept close to the original experiment code to reduce the possibility of discrepancies between the released artifact and the reported results.

## Repository contents

- `Pritabicl_Privsyn_Main.py` — main PriTabICL implementation using PrivSyn, including the PriTabICL pipeline and experiment runner.
- `Pritabicl_gem.py` — GEM-based experiments, including the global GEM baseline and the PriTabICL-GEM variant.
- `requirements.txt` — Python dependencies.

Public schema JSON files used by the experiments should be placed in `schemas/` or alongside the corresponding dataset files. The main script searches these locations automatically.

## Environment

The experiments were run in a GPU-enabled Google Colab environment. The implementation loads:

- `meta-llama/Llama-2-13b-chat-hf` for in-context classification.
- `Qwen/Qwen2.5-14B-Instruct` for feature-weight estimation.

Access to the corresponding Hugging Face model repositories is required. In Colab, authenticate before running the experiments:

```python
from huggingface_hub import login
login()
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## External repositories

### PrivSyn

The PrivSyn implementation is loaded from the original repository.

```bash
git clone https://github.com/sukjingitsit/PrivSyn.git /content/PrivSyn
export PRIVSYN_HOME=/content/PrivSyn
```

In a Colab notebook, the environment variable can instead be set with:

```python
%env PRIVSYN_HOME=/content/PrivSyn
```

### GEM

The GEM experiments use the implementation from `terranceliu/iterative-dp`:
git clone https://github.com/terranceliu/iterative-dp.git /content/iterative-dp

## Dataset organization

Dataset locations are specified in the `DATASET_META` dictionary in `Pritabicl_Privsyn_Main.py`.

The released experimental configuration uses paths of the form:

```text
/content/drive/MyDrive/Colab Notebooks/datasets/<dataset>/train.csv
/content/drive/MyDrive/Colab Notebooks/datasets/<dataset>/test.csv
```

To run the code in another environment, change the `path` and `test` entries in `DATASET_META` to the local locations of the corresponding train and test files. No other changes to the experimental pipeline are required.

For example:

```python
'Adult': {
    'path': '/path/to/datasets/adult/train.csv',
    'test': '/path/to/datasets/adult/test.csv',
    ...
}
```

Schema files can be stored in:

```text
schemas/
```

or in the corresponding dataset directory. The schema loader searches both locations.

## Running PriTabICL with PrivSyn

The main script accepts the dataset names, number of trials, cluster counts, privacy budgets, output file, and methods through command-line arguments.

Example:

```bash
python Pritabicl_Privsyn_Main.py \
  --datasets Phishing Diabetes Adult Airline \
  --trials 10 \
  --k_list 3 \
  --eps_list 1 2 4\
  --methods C3_PRIVSYN_ICL_DP_SAMPLE_LLM \
  --results_csv pritabicl_privsyn_results.csv
```

Multiple epsilon values can be supplied:

```bash
python Pritabicl_Privsyn_Main.py \
  --datasets Phishing \
  --trials 10 \
  --k_list 3 \
  --eps_list 1 2 4 \
  --methods C3_PRIVSYN_ICL_DP_SAMPLE_LLM \
  --results_csv adult_results.csv
```


## Running the GEM experiments

`Pritabicl_gem.py` imports the main PriTabICL script and the external GEM implementation. Because the released base script has been renamed, pass its path explicitly with `--base-script`.

Example for the PriTabICL-GEM variant:

```bash
python Pritabicl_gem.py \
  --base-script Pritabicl_Privsyn_Main.py \
  --gem-home /content/iterative-dp \
  --datasets Adult Phishing Airline Diabetes \
  --methods C3_GEM_ICL_DP_SAMPLE_LLM \
  --trials 10 \
  --k-list 3 \
  --eps-list 1.0 \
  --cluster-share 0.30 \
  --delta 1e-5 \
  --results-csv pritabicl_gem_results.csv
```

The global GEM baseline can be run with:

```bash
python Pritabicl_gem.py \
  --base-script Pritabicl_Privsyn_Main.py \
  --gem-home /content/iterative-dp \
  --datasets Adult \
  --methods GLOBAL_GEM \
  --trials 10 \
  --k-list 3 \
  --eps-list 1.0 \
  --delta 1e-5 \
  --results-csv global_gem_results.csv
```

## Main experimental parameters

The experiment scripts expose the principal settings used in the paper through command-line arguments:

- `--datasets` — datasets to evaluate.
- `--trials` — number of experimental trials.
- `--k_list` / `--k-list` — number of clusters for the PrivSyn / GEM runners, respectively.
- `--eps_list` / `--eps-list` — privacy budget(s).
- `--methods` — method(s) to evaluate.
- `--results_csv` / `--results-csv` — output CSV.
- `--weight_sample_frac` / `--weight-sample-frac` — disjoint fraction used for DP-summary-based feature weighting.
- `--cluster-share` — fraction of epsilon allocated to DP clustering in the GEM runner.
- `--delta` — delta used by GEM.

Results are appended to the specified CSV file.


## Reproducibility notes

The implementation uses fixed seeds for the data partitioning and method-specific randomization. GPU libraries and LLM inference can nevertheless exhibit environment-dependent numerical differences. The released scripts retain the experiment logic used for the reported results rather than being refactored into a new software architecture.
