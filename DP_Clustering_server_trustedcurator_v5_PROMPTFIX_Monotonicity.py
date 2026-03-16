import os, gc, random, argparse, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
from snsynth import Synthesizer
from snsynth.transform import (
    TableTransformer,
    MinMaxTransformer,
    ChainTransformer,
    LabelTransformer,
    OneHotEncoder,
)
# -------------------------------
# Trusted-curator threat model
# -------------------------------
THREAT_MODEL = "trusted_curator"  # This code assumes trusted curator for any non-DP preprocessing (e.g., KMeans).
DEBUG_GENERATE = False
# -------------------------------
# Dataset schema loading (public metadata)
# -------------------------------
def load_dataset_schema_json(dataset_name, schema_dir=None):
    fname_map = {
        "Adult": "adult.json",
        "Magic": "magic.json",
        "Phishing": "phishing.json",
        "Abalone": "abalone.json",
        "Shoppers": "shoppers.json",
    }
    target = fname_map.get(dataset_name, None)
    if target is None:
        return None

    if schema_dir is None:
        ds_lower = dataset_name.lower()
        candidates = [
            os.path.join(os.getcwd(), "schemas"),
            os.getcwd(),
            os.path.join(os.getcwd(), "datasets", ds_lower),
            os.path.join(os.getcwd(), "datasets", ds_lower.replace(" ", "")),
            "/mnt/data",
        ]
    else:
        candidates = [schema_dir]

    for base in candidates:
        for fname in (target, target.lower(), target.upper()):
            path = os.path.join(base, fname)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)

    return None

def append_row_to_csv(row: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_row = pd.DataFrame([row])
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    df_row.to_csv(csv_path, mode="a", header=not exists, index=False)

def schema_to_domains_and_bounds(schema):
    cat_domains = {}
    num_bounds = {}
    if not schema:
        return cat_domains, num_bounds
    for col in schema.get("columns", []):
        name = col.get("name")
        ctype = col.get("type")
        if ctype == "categorical":
            # i2s is index-to-string list
            if "i2s" in col and isinstance(col["i2s"], list):
                cat_domains[name] = list(map(str, col["i2s"]))
        elif ctype == "continuous":
            if "min" in col and "max" in col:
                num_bounds[name] = (float(col["min"]), float(col["max"]))
    return cat_domains, num_bounds

def clip_and_scale_numeric(df, num_cols, num_bounds):
    """Clip numeric columns to (L,U) and scale to [0,1] using fixed bounds (not data-derived)."""
    df = df.copy()
    for c in num_cols:
        if c not in num_bounds:
            raise ValueError(f"Missing numeric bounds for '{c}'. Provide schema JSON with min/max.")
        L, U = num_bounds[c]
        if U <= L:
            raise ValueError(f"Invalid bounds for '{c}': ({L},{U})")
        df[c] = df[c].clip(L, U)
        df[c] = (df[c] - L) / (U - L)
    return df

def unscale_numeric_value(x01, L, U):
    return float(x01) * (U - L) + L

def get_cat_domains(cfg, train_df, cat_cols, allow_private_fallback=False):
    """Return categorical domains for RR / noisy-max.
    In trusted-curator experiments we still avoid deriving domains from private data by default,
    because it complicates the DP story. Provide domains via dataset schema JSON (recommended).
    """
    domains = {}
    cfg_domains = cfg.get("cat_domains", None) if isinstance(cfg, dict) else None
    if cfg_domains:
        for c in cat_cols:
            if c in cfg_domains:
                domains[c] = list(map(str, cfg_domains[c]))

    missing = [c for c in cat_cols if c not in domains]
    if missing and not allow_private_fallback:
        raise ValueError(
            "Missing categorical domains for columns: %s. Provide schema-derived cat_domains (recommended)." % missing
        )

    # Fallback (NOT recommended): derive from private training data
    if missing and allow_private_fallback:
        print("[WARN] Falling back to train_df.unique() domains for:", missing)
        for c in missing:
            domains[c] = list(map(str, sorted(train_df[c].astype(str).unique().tolist())))
    return domains



# =========================
# CONFIG
# =========================
DATASETS_TO_RUN = ["Adult", "Magic", "Phishing", "Abalone", "Shoppers" ]

# Protocol Config
N_TRIALS = 20
SHOT_COUNTS = [10]
EVAL_PER_CLASS = 50
K_LIST = [8]
CLUSTER_SEED = 42

#["A2","B1","C1","C2a"]
METHODS = ["A2","B1","C1","C2a"]
RESULTS_CSV = "Epsilon_protocol_results.csv"
PLOTS_DIR = "plots"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# UTILITIES & DP MECHANISMS
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rr(rng, val, domain, eps: float):
    domain = list(domain)
    if len(domain) <= 1 or eps > 100:
        return val
    p = np.exp(eps) / (np.exp(eps) + len(domain) - 1)
    return val if rng.rand() < p else rng.choice([v for v in domain if v != val])

def dp_noisy_max(rng, counts: Counter, epsilon: float, possible_values):
    possible_values = list(possible_values)
    if len(possible_values) == 0:
        return None
    safe = {v: counts.get(v, 0) + rng.laplace(0, 1.0/epsilon) for v in possible_values}
    return max(safe, key=safe.get)

def mixed_dist(row_dict, cand, num_cols, cat_cols):
    d_num = 0.0
    for c in num_cols:
        d_num += abs(float(row_dict[c]) - float(cand[c]))   # L1, not squared L2

    d_cat = 0.0
    for c in cat_cols:
        d_cat += 1.0 if str(row_dict[c]) != str(cand[c]) else 0.0

    return d_num + d_cat

def compute_internal_distortion(df, clusters, dp_centroids, num_cols, cat_cols):
    distortions = []
    for c in np.unique(clusters):
        rows = df[clusters == c]
        if rows.empty or c not in dp_centroids: continue
        true_cent = {col: rows[col].mean() for col in num_cols}
        for col in cat_cols: true_cent[col] = rows[col].mode()[0]
        distortions.append(mixed_dist(true_cent, dp_centroids[c], num_cols, cat_cols))
    return float(np.mean(distortions)) if distortions else 0.0

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# DATA PREP
# =========================
DATASET_META = {
    "Adult": {
        "path": "/content/drive/MyDrive/Colab Notebooks/datasets/adult/train.csv",
        "test": "/content/drive/MyDrive/Colab Notebooks/datasets/adult/test.csv",
        "label": "label", "pos": ">50K", "task": "income prediction"
    },
    "Magic": {
        "path": "/content/drive/MyDrive/Colab Notebooks/datasets/magic/train.csv",
        "test": "/content/drive/MyDrive/Colab Notebooks/datasets/magic/test.csv",
        "label": "label", "pos": "g", "task": "particle classification"
    },
    "Phishing": {
        "path": "/content/drive/MyDrive/Colab Notebooks/datasets/phishing/train.csv",
        "test": "/content/drive/MyDrive/Colab Notebooks/datasets/phishing/test.csv",
        "label": "label", "pos": "1", "task": "phishing website detection"
    },
    "Shoppers": {
        "path": "/content/drive/MyDrive/Colab Notebooks/datasets/shoppers/train.csv",
        "test": "/content/drive/MyDrive/Colab Notebooks/datasets/shoppers/test.csv",
        "label": "label", "pos": "TRUE", "task": "online purchase prediction"
    },
    "Abalone": {
        "path": "/content/drive/MyDrive/Colab Notebooks/datasets/Abalone/train.csv",
        "test": "/content/drive/MyDrive/Colab Notebooks/datasets/Abalone/test.csv",
        "label": "label", "pos": "10", "task": "abalone age classification"
    },
}

def check_data_exists(dataset_name):
    cfg = DATASET_META[dataset_name]
    if not os.path.exists(cfg["path"]) or not os.path.exists(cfg["test"]):
        raise FileNotFoundError(
            f"\n\n🚨 ERROR: Could not find the dataset files for {dataset_name}!\n"
            f"Expected paths:\n  Train: {cfg['path']}\n  Test:  {cfg['test']}\n"
            f"--> Action Required: Please upload your 'datasets' folder to this environment.\n"
        )

def prep_dataset(df, cfg, name):
    df = df.copy().replace("?", np.nan).dropna()
    if name == "Abalone":
        df["target"] = (df[cfg["label"]].astype(float) > 9.0).astype(int)
    else:
        df["target"] = (df[cfg["label"]].astype(str).str.strip().str.upper() == str(cfg["pos"]).strip().upper()).astype(int)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for x in ["target", cfg["label"]]:
        if x in num_cols: num_cols.remove(x)
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ["target", cfg["label"]]]
    return df[num_cols + cat_cols + ["target"]].dropna(), num_cols, cat_cols

def make_balanced_eval(test_df, per_class, seed):
    pos = test_df[test_df["target"] == 1]
    neg = test_df[test_df["target"] == 0]
    n = min(len(pos), len(neg), per_class)
    if n == 0: return None
    return pd.concat([pos.sample(n, random_state=seed), neg.sample(n, random_state=seed)]).sample(frac=1, random_state=seed).reset_index(drop=True)

# =========================
# MODEL LOADING (Global)
# =========================
print(f"Loading: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

model.eval()


@torch.no_grad()
def _make_chat_prompt(system_msg: str, user_msg: str) -> str:
    """Build a chat-formatted prompt when supported by the tokenizer.
    Falls back to a plain concatenation if apply_chat_template is unavailable.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system_msg + "\n\n" + user_msg + "\nAnswer: "

def format_value(v):
    # Keep formatting stable across datasets/strategies to be fair
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NA"
    except Exception:
        pass
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    s = str(v)
    return s[:60]  # prevent very long categorical strings from inflating token count

def format_row(row_dict: dict, cols: list) -> str:
    return "\n".join([f"- {c}: {format_value(row_dict.get(c))}" for c in cols])

def build_user_msg(cfg: dict, row_dict: dict, shots_df, cols: list) -> str:
    task = cfg.get("task", "classification")
    parts = [f"Task: {task}", "Output: 1 or 0\n"]

    # Compact Few-Shot Examples
    if shots_df is not None and len(shots_df) > 0:
        for _, s in shots_df.iterrows():
            feat_list = [f"{c}: {format_value(s[c])}" for c in cols]
            feat_str = " | ".join(feat_list)
            parts.append(f"Input: {feat_str} -> Label: {int(s['target'])}")

    # Target Query
    target_feats = " | ".join([f"{c}: {format_value(row_dict[c])}" for c in cols])
    parts.append(f"\nTarget Input: {target_feats} -> Label:")
    
    return "\n".join(parts)

def score_completion(enc, completion_text: str) -> float:
    comp_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    out = model(**enc, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    logp = 0.0

    for tid in comp_ids:
        logp += torch.log_softmax(logits[0], dim=-1)[tid].item()

        inp = torch.tensor([[tid]], device=model.device)
        out = model(input_ids=inp, past_key_values=past, use_cache=True)

        past = out.past_key_values
        logits = out.logits[:, -1, :]

    return float(logp)
CAND0_STRS = ["0"]
CAND1_STRS = ["1"]


@torch.no_grad()
def llm_predict_binary(system_msg: str, user_msg: str) -> int:
    # Get Actual Scores
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_txt, return_tensors="pt", truncation=True, max_length=8100).to(model.device)
    
    # Just grab the raw log probabilities for 0 and 1
    s0 = score_completion(enc, "0")
    s1 = score_completion(enc, "1")

    # Trust the raw logits directly now that the prompt format is strictly "Input -> Label"
    pred = 1 if s1 > s0 else 0
    return pred

def retrieve_balanced_nearest_shots(pool_df, row_dict, k, num_cols, cat_cols, seed=None, label_col="target"):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None

    k = int(min(k, len(pool_df)))
    if k <= 0:
        return None

    pos = pool_df[pool_df[label_col] == 1].copy()
    neg = pool_df[pool_df[label_col] == 0].copy()

    def _add_dist(df):
        if len(df) == 0:
            return df
        out = df.copy()
        out["_dist"] = out.apply(
            lambda r: mixed_dist(
                row_dict,
                {c: r[c] for c in (num_cols + cat_cols)},
                num_cols,
                cat_cols,
            ),
            axis=1,
        )
        return out

    pos = _add_dist(pos)
    neg = _add_dist(neg)

    k_pos = k // 2
    k_neg = k - k_pos

    pos_shots = pos.sort_values("_dist").head(k_pos) if len(pos) > 0 else pos
    neg_shots = neg.sort_values("_dist").head(k_neg) if len(neg) > 0 else neg

    shots = pd.concat([pos_shots, neg_shots], axis=0)

    # if one class is short, fill from the other class by nearest remaining rows
    remaining = k - len(shots)
    if remaining > 0:
        used_idx = set(shots.index.tolist())
        leftover = pd.concat([pos, neg], axis=0)
        leftover = leftover[~leftover.index.isin(used_idx)].sort_values("_dist")
        fill = leftover.head(remaining)
        shots = pd.concat([shots, fill], axis=0)

    if seed is not None and len(shots) > 0:
        shots = shots.sample(frac=1.0, random_state=seed)

    return shots.drop(columns=["_dist"], errors="ignore").reset_index(drop=True)
def sample_balanced_shots(pool_df: pd.DataFrame, k: int, seed: int, label_col: str = "target"):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None

    k = int(min(k, len(pool_df)))
    if k <= 0:
        return None

    pos = pool_df[pool_df[label_col] == 1]
    neg = pool_df[pool_df[label_col] == 0]

    if len(pos) == 0 or len(neg) == 0:
        shots = pool_df.sample(n=k, replace=(k > len(pool_df)), random_state=seed)
        return shots.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)

    k_pos = k // 2
    k_neg = k - k_pos

    s_pos = pos.sample(n=k_pos, replace=(k_pos > len(pos)), random_state=seed)
    s_neg = neg.sample(n=k_neg, replace=(k_neg > len(neg)), random_state=seed + 1)

    shots = pd.concat([s_pos, s_neg], axis=0)
    shots = shots.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)

    print("[DEBUG shots] seed =", seed, "| labels =", shots[label_col].tolist())
    return shots

def retrieve_nearest_shots(pool_df, row_dict, k, num_cols, cat_cols, seed=None):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None

    df = pool_df.copy()

    def _dist(r):
        cand = {c: r[c] for c in (num_cols + cat_cols)}
        return mixed_dist(row_dict, cand, num_cols, cat_cols)

    df["_dist"] = df.apply(_dist, axis=1)

    # take the k most similar rows
    shots = df.sort_values("_dist").head(k).copy()

    # optional shuffle so prompt order is not always nearest-to-farthest
    if seed is not None:
        shots = shots.sample(frac=1.0, random_state=seed)

    return shots.drop(columns=["_dist"], errors="ignore").reset_index(drop=True)
# =========================
# STRATEGIES & DP ARCHITECTURES
# =========================

def build_clean_pool(train_df, k_pool=2000, seed=42):
    return train_df.copy().reset_index(drop=True) if len(train_df) <= k_pool else train_df.sample(k_pool, random_state=seed).reset_index(drop=True)

def unscale_row_dict(row_dict, num_bounds):
    """Helper to unscale a single dict row for the LLM prompt"""
    out = dict(row_dict)
    for c, (L, U) in num_bounds.items():
        if c in out:
            try:
                # Reverse the MinMax scaling: val * (U - L) + L
                val = float(out[c]) * (U - L) + L
                out[c] = round(val, 2)  # Make it clean for the LLM
            except Exception:
                pass
    return out

def execute_classification_strategy(cfg, row_dict, shots, cols):
    num_bounds = cfg.get("num_bounds", {})
    
    # 1. Unscale the Target Query
    clean_row = unscale_row_dict(row_dict, num_bounds)
    
    # 2. Unscale the Few-Shot Examples (if any)
    if shots is not None and len(shots) > 0:
        clean_shots = shots.copy()
        for c, (L, U) in num_bounds.items():
            if c in clean_shots.columns:
                # Unscale the column
                clean_shots[c] = pd.to_numeric(clean_shots[c], errors="coerce") * (U - L) + L
                clean_shots[c] = clean_shots[c].round(2)
    else:
        clean_shots = shots
        
    system_msg = "Follow the Input -> Label pattern. Output only 1 or 0. No explanation."
    
    # Pass the human-readable, unscaled data to the prompt builder
    user_msg = build_user_msg(cfg, clean_row, clean_shots, cols)
    return llm_predict_binary(system_msg, user_msg)

def strat_A2_clean_fewshot(cfg, row_dict, pool_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(pool_df))
    #shots = sample_balanced_shots(pool_df, k=k, seed=seed) if k > 0 else None
    #shots = retrieve_nearest_shots(pool_df, row_dict, k,num_cols, cat_cols,seed=seed) if k > 0 else None
    shots = retrieve_balanced_nearest_shots(pool_df, row_dict, k, num_cols, cat_cols, seed=seed) if k > 0 else None
    if shots is not None:
      print("[DEBUG A2] shot label counts:",
      shots["target"].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, (num_cols + cat_cols))

def build_B1_ldp_pool(train_df, num_cols, cat_cols, eps_total, rng, cat_domains):
    e_col = eps_total / max(len(num_cols) + len(cat_cols) + 1, 1)
    df_dp = train_df.copy()

    for c in cat_cols:
        df_dp[c] = df_dp[c].astype(str)

    # numeric perturbation
    if len(num_cols) > 0:
        noisy = df_dp[num_cols] + rng.laplace(0, 1.0 / e_col, size=df_dp[num_cols].shape)
        df_dp[num_cols] = noisy

    # categorical randomized response
    for c in cat_cols:
        df_dp[c] = df_dp[c].apply(lambda x: rr(rng, str(x), cat_domains[c], e_col))

    # label randomized response
    df_dp["target"] = df_dp["target"].apply(lambda x: rr(rng, int(x), [0, 1], e_col))

    return df_dp

def strat_B1_ldp(cfg, row_dict, dp_pool_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(dp_pool_df))
    shots = sample_balanced_shots(dp_pool_df, k=k, seed=seed) if k > 0 else None
    #shots = retrieve_nearest_shots(dp_pool_df, row_dict, k,num_cols, cat_cols,seed=seed) if k > 0 else None
    if shots is not None:
      print("[DEBUG B1] shot label counts:",
      shots["target"].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, (num_cols + cat_cols))
    
def build_C2a_clustered_dp_synth_label_conditional_2way(
    train_df, num_cols, cat_cols,
    eps_cent, eps_mem, eps_synth,
    k, rng, rows_per_cluster,
    cluster_seed=42, min_cluster_size=5,
    clusters=None, cat_domains=None
):
    """
    C2a with:
      - trusted-curator fixed clustering
      - DP centroid release
      - RR membership
      - per-cluster label-conditional synthesis
      - selected categorical two-way marginals inside each cluster
    """

    n_features = len(num_cols) + len(cat_cols)
    eps_cent = float(eps_cent)
    eps_mem = float(eps_mem)
    eps_synth = float(eps_synth)

    e_cent_per_feature = eps_cent / max(n_features, 1)

    # same trusted-curator clustering logic
    if clusters is None:
      X_num = train_df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else np.empty((len(train_df), 0), dtype=float)

      if len(cat_cols) > 0:
        X_cat = np.stack(
            [train_df[c].astype("category").cat.codes.values for c in cat_cols],
            axis=1
        ).astype(float)
      else:
        X_cat = np.empty((len(train_df), 0), dtype=float)

      X_all = np.concatenate([X_num, X_cat], axis=1)

      if X_all.shape[1] == 0:
        raise ValueError("No features available for clustering.")

      km = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=100,
        random_state=cluster_seed
      ).fit(X_all)
      clusters = km.labels_

    if cat_domains is None:
        raise ValueError("cat_domains must be provided from the public schema.")

    # DP centroids: same as current C2a
    dp_centroids = {}
    for c in np.unique(clusters):
        rows = train_df[clusters == c]
        dp_centroids[c] = {}

        for col in num_cols:
            e_sum = e_cent_per_feature / 2.0
            e_cnt = e_cent_per_feature / 2.0
            noisy_sum = float(rows[col].sum()) + rng.laplace(0, 1.0 / max(e_sum, 1e-9))
            noisy_cnt = float(len(rows)) + rng.laplace(0, 1.0 / max(e_cnt, 1e-9))
            dp_centroids[c][col] = noisy_sum / max(noisy_cnt, 1e-5)

        for col in cat_cols:
            dp_centroids[c][col] = dp_noisy_max(
                rng, Counter(rows[col].astype(str)),
                e_cent_per_feature, cat_domains[col]
            )

    distortion = compute_internal_distortion(train_df, clusters, dp_centroids, num_cols, cat_cols)
    cluster_ids = list(dp_centroids.keys())

    # RR membership: same as current C2a
    def _assign_then_rr(r):
        best = min(cluster_ids, key=lambda cid: mixed_dist(r.to_dict(), dp_centroids[cid], num_cols, cat_cols))
        return rr(rng, best, cluster_ids, eps_mem)

    mem = train_df.apply(_assign_then_rr, axis=1)
    synth_store = {}

    for c in cluster_ids:
        df_c = train_df[mem == c].copy()
        print(
          f"[DEBUG C2a real] cluster={c} size={len(df_c)} "
          f"real_labels={df_c['target'].value_counts().to_dict()}"
              )

        if len(df_c) < min_cluster_size:
         # Private fallback built only from already-private centroid info + public schema.
          centroid = dp_centroids[c]

          fallback_rows = pd.DataFrame({
           **{
            col: [float(np.clip(centroid.get(col, 0.5), 0.0, 1.0))] * rows_per_cluster
            for col in num_cols
              },
            **{
            col: [str(centroid.get(col, cat_domains[col][0]))] * rows_per_cluster
            for col in cat_cols
              },
            "target": ([0, 1] * (rows_per_cluster // 2)) + ([0] if rows_per_cluster % 2 else []),
            }).iloc[:rows_per_cluster].reset_index(drop=True)

          synth_store[c] = fallback_rows
          print(
            f"[DEBUG C2a fallback] cluster={c} using centroid-based private fallback "
            f"size={len(fallback_rows)} label_counts={fallback_rows['target'].value_counts().to_dict()}"
                )
          continue

        # Use full cluster synthesis budget for each cluster under parallel composition
        set_seed(int(rng.randint(0, 10**9)))
        synth_store[c] = dpctgan_sample_cluster(
          df_cluster=df_c,
          num_cols=num_cols,
          cat_cols=cat_cols,
          eps_cluster=eps_synth,
          n_rows=rows_per_cluster,
                                                )

        print(
            f"[DEBUG C2a synth] cluster={c} size={len(df_c)} "
            f"synth_labels={synth_store[c]['target'].value_counts().to_dict()}"
        )


    return dp_centroids, synth_store, distortion


def make_dpctgan_transformer(num_cols, cat_cols):
    transformers = []

    for _ in num_cols:
        transformers.append(MinMaxTransformer(lower=0.0, upper=1.0, nullable=False))

    for _ in cat_cols:
        transformers.append(ChainTransformer([LabelTransformer(), OneHotEncoder()]))

    transformers.append(ChainTransformer([LabelTransformer(), OneHotEncoder()]))  # target
    return TableTransformer(transformers)

def build_C1_dpctgan_pool(train_df, num_cols, cat_cols, eps_total, rng, n_synth=None):
    df = train_df.copy()

    if n_synth is None:
        n_synth = len(df)

    ordered_cols = list(num_cols) + list(cat_cols) + ["target"]
    df = df[ordered_cols].copy()

    for c in cat_cols:
        df[c] = df[c].astype(str)
    df["target"] = df["target"].astype(str)

    tt = make_dpctgan_transformer(num_cols, cat_cols)

    synth = Synthesizer.create(
        "dpctgan",
        epsilon=float(eps_total),
        verbose=False,
    )
    synth.fit(df, transformer=tt, preprocessor_eps=0.0)
    synth_df = synth.sample(n_synth).copy()

    for c in num_cols:
        synth_df[c] = pd.to_numeric(synth_df[c], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    for c in cat_cols:
        synth_df[c] = synth_df[c].astype(str)

    synth_df["target"] = (
        pd.to_numeric(synth_df["target"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )

    print("[DEBUG C1] synth size:", len(synth_df))
    print("[DEBUG C1] synth label counts:", synth_df["target"].value_counts().to_dict())
    return synth_df.reset_index(drop=True)

def dpctgan_sample_cluster(df_cluster, num_cols, cat_cols, eps_cluster, n_rows):
    ordered_cols = list(num_cols) + list(cat_cols) + ["target"]
    df = df_cluster[ordered_cols].copy()

    for c in cat_cols:
        df[c] = df[c].astype(str)
    df["target"] = df["target"].astype(str)

    tt = make_dpctgan_transformer(num_cols, cat_cols)

    synth = Synthesizer.create(
        "dpctgan",
        epsilon=float(eps_cluster),
        verbose=False,
    )
    synth.fit(df, transformer=tt, preprocessor_eps=0.0)
    out = synth.sample(n_rows).copy()

    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    for c in cat_cols:
        out[c] = out[c].astype(str)

    out["target"] = (
        pd.to_numeric(out["target"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )
    return out.reset_index(drop=True)


def strat_C1_global_synth(cfg, row_dict, synth_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(synth_df))
    shots = sample_balanced_shots(synth_df, k=k, seed=seed) if k > 0 else None
    #shots = retrieve_nearest_shots(synth_df, row_dict, k, num_cols, cat_cols, seed=seed) if k > 0 else None
    #shots = synth_df.sample(n=k, random_state=seed, replace=(k > len(synth_df))).reset_index(drop=True) if k > 0 else None
    if shots is not None:
      print("[DEBUG C1] shot label counts:",
      shots["target"].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, (num_cols + cat_cols))

def strat_C2a_clustered_synth(cfg, row_dict, dp_centroids, synth_store, num_cols, cat_cols, k_retr, seed):
    # choose the single nearest privatized cluster
    dists = []
    for cid in dp_centroids.keys():
        d = mixed_dist(row_dict, dp_centroids[cid], num_cols, cat_cols)
        dists.append((cid, d))
    dists.sort(key=lambda x: x[1])

    best_cluster = dists[0][0]
    pool = synth_store[best_cluster].reset_index(drop=True)

    k = min(k_retr, len(pool))
    #shots = retrieve_nearest_shots(pool, row_dict, k, num_cols, cat_cols, seed=seed) if k > 0 else None
    shots = sample_balanced_shots(pool, k=k, seed=seed) if k > 0 else None
    #shots = pool.sample(n=k, random_state=seed, replace=(k > len(pool))).reset_index(drop=True) if k > 0 else None
    if shots is not None:
      print(
        "[DEBUG C2a] seed:", seed,
        "| best cluster:", best_cluster,
        "| shot label counts:", shots["target"].value_counts().to_dict(),
        "| first rows:", shots.head(2).to_dict("records")
          )
    if shots is None or len(shots) == 0:
        return 0
    return execute_classification_strategy(cfg, row_dict, shots, (num_cols + cat_cols))

def run_epsilon_protocol(
    datasets_to_run=None, 
    n_trials=20, 
    k_list=None, 
    eps_list=None, 
    results_csv="epsilon_protocol_results.csv"
):
    datasets = datasets_to_run or DATASETS_TO_RUN
    k_values = k_list or K_LIST
    eps_values = eps_list or [0.1, 0.5, 1.0, 2.0, 5.0] # The Epsilon Sweep
    system_msg = "Follow the Input -> Label pattern. Output only 1 or 0. No explanation."

    for dataset_name in datasets:
        check_data_exists(dataset_name)
        cfg = DATASET_META[dataset_name]
        print(f"\n=== DATASET: {dataset_name} ===")

        # --- 1. DATA LOADING & PREP (Once per dataset) ---
        train_raw = pd.read_csv(cfg["path"])
        test_raw  = pd.read_csv(cfg["test"])

        train_df, num_cols, cat_cols = prep_dataset(train_raw, cfg, dataset_name)
        test_df,  _, _               = prep_dataset(test_raw,  cfg, dataset_name)

        schema = load_dataset_schema_json(dataset_name)
        cat_domains, num_bounds = schema_to_domains_and_bounds(schema)
        cfg = dict(cfg)
        cfg["name"] = dataset_name
        cfg["cat_domains"] = cat_domains
        cfg["num_bounds"] = num_bounds

        feat_cols = [c for c in train_df.columns if c != "target"]
        num_cols = [c for c in feat_cols if c in num_bounds]
        cat_cols = [c for c in feat_cols if c in cat_domains and c not in num_cols]

        for c in cat_cols:
            train_df[c] = train_df[c].astype(str)
            test_df[c]  = test_df[c].astype(str)

        if len(num_cols) > 0:
            train_df = clip_and_scale_numeric(train_df, num_cols, num_bounds)
            test_df  = clip_and_scale_numeric(test_df,  num_cols, num_bounds)

        eval_df = make_balanced_eval(test_df, per_class=EVAL_PER_CLASS, seed=777)
        if eval_df is None or len(eval_df) == 0:
            continue

        y_true = eval_df["target"].astype(int).tolist()
        clean_pool = build_clean_pool(train_df, k_pool=min(5000, len(train_df)), seed=42)

        # --- 2. CLUSTERING & SWEEP LOOPS ---
        for K in k_values:
            
            # Feature matrix for K-Means (Trusted Curator)
            X_num = train_df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
            X_cat = np.stack([train_df[c].astype("category").cat.codes.values for c in cat_cols], axis=1).astype(float) if len(cat_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
            X_all = np.concatenate([X_num, X_cat], axis=1)

            clusters = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=CLUSTER_SEED).fit(X_all).labels_

            for eps in eps_values:
                print(f"\n>>> K={K} | Epsilon={eps} <<<")
                
                # Dynamic C2a budget split for the current epsilon
                eps_c2a_cent = 0.30 * eps
                eps_c2a_mem  = 0.20 * eps
                eps_c2a_syn  = 0.50 * eps

                for trial_seed in [1000 + i for i in range(n_trials)]:
                    
                    # --- 3. SYNTHESIZE DP DATA ---
                    rng_b1 = np.random.RandomState(trial_seed + 101)
                    rng_c1 = np.random.RandomState(trial_seed + 202)
                    rng_c2 = np.random.RandomState(trial_seed + 303)

                    b1_pool = build_B1_ldp_pool(train_df, num_cols, cat_cols, eps, rng_b1, cat_domains)
                    
                    set_seed(trial_seed + 202)
                    c1_synth = build_C1_dpctgan_pool(train_df, num_cols, cat_cols, eps, rng_c1, n_synth=len(train_df))
                    
                    c2_cent, c2_store, _ = build_C2a_clustered_dp_synth_label_conditional_2way(
                        train_df, num_cols, cat_cols, 
                        eps_c2a_cent, eps_c2a_mem, eps_c2a_syn,
                        K, rng_c2, len(train_df) // max(K, 1),
                        clusters=clusters, cat_domains=cat_domains
                    )

                    # --- 4. EVALUATE STRATEGIES ---
                    for method in METHODS:
                        preds = []
                        shots_used = SHOT_COUNTS[0]

                        for ridx, (_, row) in enumerate(eval_df.iterrows()):
                            row_dict = {col: row[col] for col in (num_cols + cat_cols)}
                            row_seed = trial_seed * 10000 + ridx

                            if method == "A2":
                                shots = retrieve_balanced_nearest_shots(clean_pool, row_dict, shots_used, num_cols, cat_cols, seed=row_seed)
                            elif method == "B1":
                                shots = sample_balanced_shots(b1_pool, shots_used, seed=row_seed)
                            elif method == "C1":
                                shots = sample_balanced_shots(c1_synth, shots_used, seed=row_seed)
                            elif method == "C2a":
                                dists = [(cid, mixed_dist(row_dict, c2_cent[cid], num_cols, cat_cols)) for cid in c2_cent.keys()]
                                best_c = min(dists, key=lambda x: x[1])[0]
                                shots = sample_balanced_shots(c2_store[best_c], shots_used, seed=row_seed)

                            # Execution handles system_msg, user_msg, and llm_predict_binary calibration
                            pred = execute_classification_strategy(cfg, row_dict, shots, (num_cols + cat_cols))
                            preds.append(int(pred))

                        f1 = f1_score(y_true, preds)
                        acc = accuracy_score(y_true, preds)
                        prec = precision_score(y_true, preds, zero_division=0)
                        rec = recall_score(y_true, preds, zero_division=0)

                        row_out = {
                            "dataset": dataset_name,
                            "K": K,
                            "epsilon": eps,
                            "shots": shots_used,
                            "seed": trial_seed,
                            "method": method,
                            "f1": f1,
                            "acc": acc,
                            "precision": prec,
                            "recall": rec
                        }
                        append_row_to_csv(row_out, results_csv)
                        print(f"[SAVED] {dataset_name} | eps={eps} | K={K} | seed={trial_seed} | method={method} | f1={f1:.3f}")

                    cleanup()

def main():
    parser = argparse.ArgumentParser(description="DP-GAN server runner")
    parser.add_argument("--mode", choices=["epsilon", "run_all"], default="epsilon")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--k_list", nargs="*", type=int, default=[8])
    parser.add_argument("--eps_list", nargs="*", type=float, default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--results_csv", default="epsilon_protocol_results.csv")
    args = parser.parse_args()

    df = run_epsilon_protocol(
        datasets_to_run=args.datasets,
        n_trials=args.trials,
        k_list=args.k_list,
        eps_list=args.eps_list,
        results_csv=args.results_csv,
    )
    
if __name__ == "__main__":
    main()
