## Overview
This repository contains the evaluation pipeline for our structure-aware differentially private synthetic memory experiments. The code evaluates and compares the utility of global DP synthesis (C1) versus clustered DP synthesis (C2a) for LLM in-context learning.

Prerequisites & Setup
Environment: Ensure you have an environment with PyTorch and the transformers library installed (BitsAndBytes is required for 4-bit quantization).
!pip -q install numpy pandas scikit-learn tqdm transformers accelerate
!pip -q install torch
!pip -q install -U transformers accelerate bitsandbytes

Dataset Paths: By default, the script looks for the datasets in a datasets/ directory.

Hugging Face Access: The script uses mistralai/Mistral-7B-Instruct-v0.2. 


## Run Command
python DP_GAN_server_neurips_trustedcurator_paperaligned_v5_PROMPTFIX_Monotonicity.py \
  --mode epsilon \
  --datasets Adult \
  --trials 50 \
  --k_list 8 \
  --results_csv "./results/eps_sweep_Adult50_0.5C1C2aa_Mono.csv"


  ## Command Line Arguments
  --mode: Set to epsilon to run the targeted DP evaluation protocol.
  --datasets: The name of the dataset to evaluate (e.g., Adult, Magic, Phishing). You can pass multiple separated by spaces.
  --trials: Number of independent trials to run (controls random seeds for DP noise and LLM sampling).
  --k_list: The number of clusters ($K$) to use for the C2a clustered synthesis strategy.
  --results_csv: The output file path where the metrics (Accuracy, F1, etc.) will be appended.