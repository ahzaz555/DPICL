

import os, gc, random, argparse, re, json
import numpy as np
import pandas as pd
import torch
import re
import torch.nn.functional as F
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    for base in candidates:
        path = os.path.join(base, target)
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
    because it complicates the DP story. Provide domains via dataset schema JSON.
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

    # Fallback Will remove it finally: derive from private training data
    if missing and allow_private_fallback:
        print("[WARN] Falling back to train_df.unique() domains for:", missing)
        for c in missing:
            domains[c] = list(map(str, sorted(train_df[c].astype(str).unique().tolist())))
    return domains



# =========================
# CONFIG
# =========================
DATASETS_TO_RUN = ["Adult", "Magic", "Phishing", "Abalone", "Shoppers" ]
EPS_TOTAL = 10

# Protocol Config
N_TRIALS = 50
TRIAL_SEEDS = [10 + i for i in range(N_TRIALS)]
SHOT_COUNTS = [10]
EVAL_PER_CLASS = 5
K_LIST = [8]
CLUSTER_SEED = 42

#["A2","B1","C1","C2a"]
METHODS = ["C1","C2a"]
RESULTS_CSV = "advisor_protocol_results.csv"
PLOTS_DIR = "plots"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# C2a epsilon split (must sum to EPS_TOTAL)
EPS_C2A_CENT = 0.30 * EPS_TOTAL
EPS_C2A_MEM  = 0.20 * EPS_TOTAL
EPS_C2A_SYN  = 0.50 * EPS_TOTAL
assert abs((EPS_C2A_CENT + EPS_C2A_MEM + EPS_C2A_SYN) - EPS_TOTAL) < 1e-9

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

def mixed_dist(row_dict, cent, num_cols, cat_cols):
    d_num = sum((float(row_dict[c]) - float(cent[c]))**2 for c in num_cols)
    d_cat = sum(1 for c in cat_cols if str(row_dict[c]) != str(cent[c]))
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
        "path": "datasets/adult/train.csv",
        "test": "datasets/adult/test.csv",
        "label": "label", "pos": ">50K", "task": "income prediction"
    },
    "Magic": {
        "path": "datasets/magic/train.csv",
        "test": "datasets/magic/test.csv",
        "label": "label", "pos": "g", "task": "particle classification"
    },
    "Phishing": {
        "path": "datasets/phishing/train.csv",
        "test": "datasets/phishing/test.csv",
        "label": "label", "pos": "1", "task": "phishing website detection"
    },
    "Shoppers": {
        "path": "shoppers/train.csv",
        "test": "shoppers/test.csv",
        "label": "label", "pos": "TRUE", "task": "online purchase prediction"
    },
    "Abalone": {
        "path": "datasets/Abalone/train.csv",
        "test": "datasets/Abalone/test.csv",
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

TOK0 = tokenizer.encode("0", add_special_tokens=False)
TOK1 = tokenizer.encode("1", add_special_tokens=False)
TOK0S = tokenizer.encode(" 0", add_special_tokens=False)  
TOK1S = tokenizer.encode(" 1", add_special_tokens=False)

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
    """Dataset-agnostic, strategy-agnostic prompt body.
    All strategies MUST call this for a fair comparison.
    """
    task = cfg.get("task", "classification")
    # Describe positive class in a dataset-agnostic way
    if cfg.get("name") == "Abalone":
        pos_desc = "label > 9 (older abalone)"
    else:
        pos_desc = str(cfg.get("pos", "positive class"))

    parts = []
    parts.append(f"Task: {task}")
    parts.append("Label definition:")
    parts.append("Important: The classes are balanced. Do NOT default to 0.")
    parts.append("If unsure, choose the class that best matches the examples.")
    parts.append(f"- Output 1 if the true label is: {pos_desc}")
    parts.append("- Output 0 otherwise")
    parts.append("")

    if shots_df is not None and len(shots_df) > 0:
        parts.append("Examples:")
        for _, s in shots_df.iterrows():
            ex = {c: s[c] for c in cols}
            parts.append("Example:")
            parts.append(format_row(ex, cols))
            parts.append(f"Label: {int(s['target'])}")
            parts.append("")

    parts.append("Target:")
    parts.append(format_row(row_dict, cols))
    parts.append("")
    parts.append("Answer:")
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
    messages = [
        {"role": "system", "content": system_msg + "\nOutput exactly one of: 0 or 1."},
        {"role": "user", "content": user_msg},
    ]

    prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_txt, return_tensors="pt", truncation=True, max_length=8100).to(model.device)

    # Optional debug generate 
    if DEBUG_GENERATE:
        gen_out = model.generate(
            **enc,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(gen_out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[DEBUG GEN] '{gen_text[:60]}'")

    s0 = max(score_completion(enc, s) for s in CAND0_STRS)
    s1 = max(score_completion(enc, s) for s in CAND1_STRS)

    pred = 1 if s1 > s0 else 0
    # print(f"[DEBUG LLM] best0={s0:.3f} best1={s1:.3f} pred={pred}")
    return pred
def sample_balanced_shots(pool_df: pd.DataFrame, k: int, seed: int, label_col: str = "target"):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None

    k = int(min(k, len(pool_df)))  # keep consistent with existing min(k, len(pool))
    rng = np.random.RandomState(seed)

    pos = pool_df[pool_df[label_col] == 1]
    neg = pool_df[pool_df[label_col] == 0]

    k_pos_t = k // 2
    k_neg_t = k - k_pos_t

    # sample without replacement if possible, else with replacement
    def _samp(df, n, rs):
        if n <= 0 or len(df) == 0:
            return df.iloc[0:0]
        rep = n > len(df)
        return df.sample(n=n, replace=rep, random_state=rs)

    s_pos = _samp(pos, min(k_pos_t, len(pos)) if len(pos) > 0 else 0, seed)
    s_neg = _samp(neg, min(k_neg_t, len(neg)) if len(neg) > 0 else 0, seed + 1)

    shots = pd.concat([s_pos, s_neg], axis=0)

    # fill remainder from the leftover pool (any class), avoiding duplicates if possible
    remaining = k - len(shots)
    if remaining > 0:
        leftover = pool_df.drop(index=shots.index, errors="ignore")
        rep = remaining > len(leftover)
        fill = leftover.sample(n=remaining, replace=rep, random_state=seed + 2) if len(leftover) > 0 else pool_df.sample(n=remaining, replace=True, random_state=seed + 2)
        shots = pd.concat([shots, fill], axis=0)

    # final shuffle
    shots = shots.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    print("[DEBUG shots] seed =", seed, "| idx =", shots.index.tolist())
    print("[DEBUG shots] seed =", seed, "| labels =", shots["target"].tolist())
    return shots
# =========================
# STRATEGIES & DP ARCHITECTURES
# =========================

def build_clean_pool(train_df, k_pool=2000, seed=42):
    return train_df.copy().reset_index(drop=True) if len(train_df) <= k_pool else train_df.sample(k_pool, random_state=seed).reset_index(drop=True)

def strat_A2_clean_fewshot(cfg, row_dict, pool_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(pool_df))
    shots = sample_balanced_shots(pool_df, k=k, seed=seed) if k > 0 else None
    if shots is not None:
      print("[DEBUG A2] shot label counts:",
      shots["target"].value_counts().to_dict())
    cols = num_cols + cat_cols
    system_msg = (
                "You are a robotic binary classifier.\n"
                "Output EXACTLY one character: 0 or 1.\n"
               "No punctuation. No explanation."
                  )
    user_msg = build_user_msg(cfg, row_dict, shots, cols)
    pred = llm_predict_binary(system_msg, user_msg)
    return pred

def build_B1_ldp_pool(train_df, num_cols, cat_cols, eps_total, rng, cat_domains):
    e_col = eps_total / max(len(num_cols) + len(cat_cols) + 1, 1)

    df_dp = train_df.copy()

    for c in cat_cols:
        df_dp[c] = df_dp[c].astype(str)

    # numeric perturbation
    if len(num_cols) > 0:
        noisy = df_dp[num_cols] + rng.laplace(0, 1.0 / e_col, size=df_dp[num_cols].shape)
        df_dp[num_cols] = np.clip(noisy, 0.0, 1.0)

    # categorical randomized response
    for c in cat_cols:
        df_dp[c] = df_dp[c].apply(lambda x: rr(rng, str(x), cat_domains[c], e_col))

    # label randomized response
    df_dp["target"] = df_dp["target"].apply(lambda x: rr(rng, int(x), [0, 1], e_col))

    return df_dp

def strat_B1_ldp(cfg, row_dict, dp_pool_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(dp_pool_df))
    shots = sample_balanced_shots(dp_pool_df, k=k, seed=seed) if k > 0 else None
    if shots is not None:
      print("[DEBUG B1] shot label counts:",
      shots["target"].value_counts().to_dict())
    cols = num_cols + cat_cols
    system_msg = (
                "You are a robotic binary classifier.\n"
                "Output EXACTLY one character: 0 or 1.\n"
               "No punctuation. No explanation."
                  )
    user_msg = build_user_msg(cfg, row_dict, shots, cols)
    pred = llm_predict_binary(system_msg, user_msg)
    return pred

def dp_synth_fit(rng, df, num_cols, cat_cols, eps_synth, cat_domains=None):
    """Fit a simple DP synthetic generator by releasing noisy sufficient statistics.

     Accounting notes:
    - Assumes numeric features are bounded/clipped to [0,1] ( pipeline min-max scales).
    - Uses add/remove adjacency.
    - Splits eps_synth across: label prior + numeric means/variances + categorical histograms.
    - Splits further across statistics (e.g., mean vs variance) to respect sequential composition.
    - cat_domains should be PUBLIC (schema-known). If not provided, caller may pass a dict.
    """

    n = max(len(df), 1)
    mu, var, cat_probs = {}, {}, {}

    eps_synth = float(eps_synth)

    eps_y   = 0.2 * eps_synth              # label prior
    eps_num = 0.5 * eps_synth              # numeric stats
    eps_cat = max(eps_synth - eps_y - eps_num, 0.0)

    eps_y = max(eps_y, 1e-9)
    eps_num = max(eps_num, 1e-9)
    eps_cat = max(eps_cat, 1e-9)


    # OPTIMIZED: The L1 sensitivity of a histogram under add/remove is 1.
    # No need to halve the epsilon!
    cts = Counter(df["target"])

    n1_noisy = max(cts.get(1, 0) + rng.laplace(0, 1.0 / eps_y), 1e-9)
    n0_noisy = max(cts.get(0, 0) + rng.laplace(0, 1.0 / eps_y), 1e-9)
    py = float(n1_noisy / (n0_noisy + n1_noisy))

    # --- Numeric stats: mean + variance ---
    if num_cols:
        p = max(len(num_cols), 1)
        eps_per_num = eps_num / p
        eps_mu = eps_per_num / 2.0
        eps_va = eps_per_num / 2.0

        for c in num_cols:
            m = float(df[c].mean()) if n > 0 else 0.5
            v = float(df[c].var(ddof=0)) if n > 0 else (1.0/12.0)

            m_noisy = m + rng.laplace(0, 1.0/(n * eps_mu))   # mean sensitivity 1/n
            v_noisy = v + rng.laplace(0, 1.0/(n * eps_va))   # heuristic; will consider DP second moment for strictness

            mu[c] = float(np.clip(m_noisy, 0.0, 1.0))
            var[c] = float(np.clip(v_noisy, 1e-6, 0.30))

    # --- Categorical histograms ---
    if cat_cols:
        q = max(len(cat_cols), 1)
        eps_per_cat = eps_cat / q

        for c in cat_cols:
            vals = list(map(str, (cat_domains.get(c, []) if cat_domains else [])))
            if not vals:
                vals = list(map(str, sorted(df[c].astype(str).unique().tolist())))  # fallback (declare public if used)

            counts = Counter(df[c].astype(str))
            noisy = np.array([counts.get(v, 0) + rng.laplace(0, 1.0/eps_per_cat) for v in vals], dtype=float)
            noisy = np.clip(noisy, 1e-9, None)
            cat_probs[c] = (vals, noisy / noisy.sum())

    return {"mu": mu, "var": var, "cat_probs": cat_probs, "py": py}

def dp_synth_sample(rng, synth_model, num_cols, cat_cols, n_rows: int):
    out = []
    for _ in range(n_rows):
        row = {}
        for c in num_cols: row[c] = float(np.clip(rng.normal(synth_model["mu"][c], np.sqrt(synth_model["var"][c])), 0.0, 1.0))
        for c in cat_cols: row[c] = str(rng.choice(synth_model["cat_probs"][c][0], p=synth_model["cat_probs"][c][1]))
        row["target"] = int(rng.rand() < synth_model["py"])
        out.append(row)
    return pd.DataFrame(out)
    
# =========================
# C1: Label-conditional + selected two-way marginals
# =========================

def dp_class_prior(rng, df, eps_y):
    """
    DP estimate of P(Y=1) using noisy class counts.
    """
    cts = Counter(df["target"])
    n1_noisy = max(cts.get(1, 0) + rng.laplace(0, 1.0 / max(eps_y, 1e-9)), 1e-9)
    n0_noisy = max(cts.get(0, 0) + rng.laplace(0, 1.0 / max(eps_y, 1e-9)), 1e-9)
    return float(n1_noisy / (n0_noisy + n1_noisy))


def dp_feature_model_fit_with_pairs(
    rng, df, num_cols, cat_cols, eps_feat, cat_domains=None, two_way_pairs=None
):
    """
    DP fit of:
      - numeric one-way marginals (mean + second moment)
      - categorical one-way marginals
      - selected categorical two-way marginals

    Numeric features are assumed already clipped/scaled to [0,1].
    """
    n = max(len(df), 1)
    mu, var, cat_probs = {}, {}, {}
    pair_probs = {}

    eps_feat = float(max(eps_feat, 1e-9))
    two_way_pairs = two_way_pairs or []

    # Split feature budget
    eps_num = 0.45 * eps_feat
    eps_cat_1way = 0.30 * eps_feat
    eps_cat_2way = max(eps_feat - eps_num - eps_cat_1way, 1e-9)

    # ---------- Numeric one-way ----------
    if num_cols:
        p = max(len(num_cols), 1)
        eps_per_num = eps_num / p
        eps_mu = eps_per_num / 2.0
        eps_m2 = eps_per_num / 2.0

        for c in num_cols:
            x = df[c].astype(float).to_numpy() if len(df) > 0 else np.array([0.5], dtype=float)
            m = float(np.mean(x))
            m2 = float(np.mean(x ** 2))

            m_noisy = m + rng.laplace(0, 1.0 / (n * max(eps_mu, 1e-9)))
            m2_noisy = m2 + rng.laplace(0, 1.0 / (n * max(eps_m2, 1e-9)))

            m_noisy = float(np.clip(m_noisy, 0.0, 1.0))
            m2_noisy = float(np.clip(m2_noisy, 0.0, 1.0))
            v_noisy = max(m2_noisy - (m_noisy ** 2), 1e-6)

            mu[c] = m_noisy
            var[c] = float(np.clip(v_noisy, 1e-6, 0.30))

    # ---------- Categorical one-way ----------
    if cat_cols:
        q = max(len(cat_cols), 1)
        eps_per_cat = eps_cat_1way / q

        for c in cat_cols:
            vals = list(map(str, (cat_domains.get(c, []) if cat_domains else [])))
            if not vals:
                raise ValueError(f"Public domain missing for categorical column '{c}'")

            counts = Counter(df[c].astype(str))
            noisy = np.array(
                [counts.get(v, 0) + rng.laplace(0, 1.0 / max(eps_per_cat, 1e-9)) for v in vals],
                dtype=float
            )
            noisy = np.clip(noisy, 1e-9, None)
            cat_probs[c] = (vals, noisy / noisy.sum())

    # ---------- Selected categorical two-way ----------
    if two_way_pairs:
        eps_per_pair = eps_cat_2way / max(len(two_way_pairs), 1)

        for c1, c2 in two_way_pairs:
            vals1 = list(map(str, cat_domains[c1]))
            vals2 = list(map(str, cat_domains[c2]))

            table = np.zeros((len(vals1), len(vals2)), dtype=float)
            i1 = {v: i for i, v in enumerate(vals1)}
            i2 = {v: j for j, v in enumerate(vals2)}

            for _, row in df[[c1, c2]].astype(str).iterrows():
                v1, v2 = row[c1], row[c2]
                if v1 in i1 and v2 in i2:
                    table[i1[v1], i2[v2]] += 1.0

            noise = rng.laplace(0, 1.0 / max(eps_per_pair, 1e-9), size=table.shape)
            noisy = np.clip(table + noise, 1e-9, None)
            joint = noisy / noisy.sum()

            pair_probs[(c1, c2)] = {
                "vals1": vals1,
                "vals2": vals2,
                "joint": joint
            }

    return {
        "mu": mu,
        "var": var,
        "cat_probs": cat_probs,
        "pair_probs": pair_probs
    }


def _sample_pair_joint(rng, pair_info):
    vals1 = pair_info["vals1"]
    vals2 = pair_info["vals2"]
    joint = pair_info["joint"]

    flat = joint.reshape(-1)
    idx = int(rng.choice(len(flat), p=flat))
    i = idx // len(vals2)
    j = idx % len(vals2)
    return vals1[i], vals2[j]


def sample_from_feature_model_with_pairs(
    rng, feature_model, num_cols, cat_cols, n_rows, label_value, two_way_pairs=None
):
    """
    Sample X|Y using:
      - selected pairwise joint draws for anchor categorical pairs
      - one-way marginals for everything else
    """
    two_way_pairs = two_way_pairs or []
    out = []

    pair_map = feature_model.get("pair_probs", {})

    # columns already set by sampled pairs
    covered_cols = set()
    for a, b in two_way_pairs:
        covered_cols.add(a)
        covered_cols.add(b)

    for _ in range(n_rows):
        row = {}

        # numeric
        for c in num_cols:
            row[c] = float(np.clip(
                rng.normal(feature_model["mu"][c], np.sqrt(feature_model["var"][c])),
                0.0, 1.0
            ))

        # sampled selected pairs jointly
        for pair in two_way_pairs:
            if pair in pair_map:
                v1, v2 = _sample_pair_joint(rng, pair_map[pair])
                row[pair[0]] = str(v1)
                row[pair[1]] = str(v2)

        # remaining categoricals independently
        for c in cat_cols:
            if c in row:
                continue
            vals, probs = feature_model["cat_probs"][c]
            row[c] = str(rng.choice(vals, p=probs))

        row["target"] = int(label_value)
        out.append(row)

    return pd.DataFrame(out)
def build_C2a_clustered_dp_synth_label_conditional_2way(
    train_df, num_cols, cat_cols,
    eps_cent, eps_mem, eps_synth,
    k, rng, rows_per_cluster,
    cluster_seed=42, min_cluster_size=5,
    clusters=None, cat_domains=None, two_way_pairs=None
):
    """
    C2a with:
      - trusted-curator fixed clustering
      - DP centroid release
      - RR membership
      - per-cluster label-conditional synthesis
      - selected categorical two-way marginals inside each cluster
    """
    two_way_pairs = two_way_pairs or []

    n_features = len(num_cols) + len(cat_cols)
    eps_cent = float(eps_cent)
    eps_mem = float(eps_mem)
    eps_synth = float(eps_synth)

    e_cent_per_feature = eps_cent / max(n_features, 1)

    # same trusted-curator clustering logic
    if clusters is None:
        if len(num_cols) > 0:
            km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=cluster_seed).fit(train_df[num_cols])
            clusters = km.labels_
        else:
            if len(cat_cols) > 0:
                X_cat = np.stack([train_df[c].astype('category').cat.codes.values for c in cat_cols], axis=1)
            else:
                X_cat = np.empty((len(train_df), 0), dtype=int)
            X_num = train_df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
            X_all = np.concatenate([X_num, X_cat], axis=1)
            if X_all.shape[1] == 0:
                raise ValueError("No features available for clustering.")
            km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=cluster_seed).fit(X_all)
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

    # new: label-conditional + two-way synth inside each privatized cluster
    synth_store = {}

    eps_y = 0.20 * eps_synth
    eps_feat = 0.80 * eps_synth

    def _public_fallback_model():
        return {
            "mu": {c: 0.5 for c in num_cols},
            "var": {c: 1.0 / 12.0 for c in num_cols},
            "cat_probs": {
                c: (
                    cat_domains[c],
                    np.ones(len(cat_domains[c]), dtype=float) / max(len(cat_domains[c]), 1)
                )
                for c in cat_cols
            },
            "pair_probs": {
                (a, b): {
                    "vals1": list(map(str, cat_domains[a])),
                    "vals2": list(map(str, cat_domains[b])),
                    "joint": np.ones((len(cat_domains[a]), len(cat_domains[b])), dtype=float)
                             / max(len(cat_domains[a]) * len(cat_domains[b]), 1)
                }
                for (a, b) in two_way_pairs
            }
        }

    for c in cluster_ids:
        df_c = train_df[mem == c].copy()

        if len(df_c) < min_cluster_size:
            n1 = rows_per_cluster // 2
            n0 = rows_per_cluster - n1
            fallback_model = _public_fallback_model()

            synth0 = sample_from_feature_model_with_pairs(
                rng, fallback_model, num_cols, cat_cols, n0,
                label_value=0, two_way_pairs=two_way_pairs
            )
            synth1 = sample_from_feature_model_with_pairs(
                rng, fallback_model, num_cols, cat_cols, n1,
                label_value=1, two_way_pairs=two_way_pairs
            )
            synth_store[c] = pd.concat([synth0, synth1], axis=0).sample(frac=1.0, random_state=321).reset_index(drop=True)
            continue

        py_c = dp_class_prior(rng, df_c, eps_y)

        df_c0 = df_c[df_c["target"] == 0].copy()
        df_c1 = df_c[df_c["target"] == 1].copy()

        model0 = (
            dp_feature_model_fit_with_pairs(
                rng, df_c0, num_cols, cat_cols, eps_feat,
                cat_domains=cat_domains, two_way_pairs=two_way_pairs
            )
            if len(df_c0) >= min_cluster_size else _public_fallback_model()
        )

        model1 = (
            dp_feature_model_fit_with_pairs(
                rng, df_c1, num_cols, cat_cols, eps_feat,
                cat_domains=cat_domains, two_way_pairs=two_way_pairs
            )
            if len(df_c1) >= min_cluster_size else _public_fallback_model()
        )

        # choose one:
        # A) DP prior-based cluster pool
        n1 = int(round(rows_per_cluster * py_c))
        n1 = min(max(n1, 0), rows_per_cluster)
        n0 = rows_per_cluster - n1

        # B) balanced cluster pool
        # n1 = rows_per_cluster // 2
        # n0 = rows_per_cluster - n1

        synth0 = sample_from_feature_model_with_pairs(
            rng, model0, num_cols, cat_cols, n0,
            label_value=0, two_way_pairs=two_way_pairs
        )
        synth1 = sample_from_feature_model_with_pairs(
            rng, model1, num_cols, cat_cols, n1,
            label_value=1, two_way_pairs=two_way_pairs
        )

        synth_store[c] = pd.concat([synth0, synth1], axis=0).sample(frac=1.0, random_state=321).reset_index(drop=True)

    return dp_centroids, synth_store, distortion

def build_C1_global_dp_synth_label_conditional_2way(
    train_df, num_cols, cat_cols, eps_total, rng, n_synth, cat_domains, two_way_pairs=None
):
    """
    Global C1 with:
      - DP class prior
      - class-conditional feature models
      - selected categorical two-way marginals inside each class
    """
    two_way_pairs = two_way_pairs or []

    eps_total = float(eps_total)
    eps_y = 0.20 * eps_total
    eps_feat = 0.80 * eps_total

    py = dp_class_prior(rng, train_df, eps_y)

    df0 = train_df[train_df["target"] == 0].copy()
    df1 = train_df[train_df["target"] == 1].copy()

    def _public_fallback_model():
        return {
            "mu": {c: 0.5 for c in num_cols},
            "var": {c: 1.0 / 12.0 for c in num_cols},
            "cat_probs": {
                c: (
                    cat_domains[c],
                    np.ones(len(cat_domains[c]), dtype=float) / max(len(cat_domains[c]), 1)
                )
                for c in cat_cols
            },
            "pair_probs": {
                (a, b): {
                    "vals1": list(map(str, cat_domains[a])),
                    "vals2": list(map(str, cat_domains[b])),
                    "joint": np.ones((len(cat_domains[a]), len(cat_domains[b])), dtype=float)
                             / max(len(cat_domains[a]) * len(cat_domains[b]), 1)
                }
                for (a, b) in two_way_pairs
            }
        }

    model0 = (
        dp_feature_model_fit_with_pairs(
            rng, df0, num_cols, cat_cols, eps_feat,
            cat_domains=cat_domains, two_way_pairs=two_way_pairs
        )
        if len(df0) >= 5 else _public_fallback_model()
    )

    model1 = (
        dp_feature_model_fit_with_pairs(
            rng, df1, num_cols, cat_cols, eps_feat,
            cat_domains=cat_domains, two_way_pairs=two_way_pairs
        )
        if len(df1) >= 5 else _public_fallback_model()
    )

    # keep the same DP class-prior logic as before
    n1 = int(round(n_synth * py))
    n1 = min(max(n1, 0), n_synth)
    n0 = n_synth - n1

    synth0 = sample_from_feature_model_with_pairs(
        rng, model0, num_cols, cat_cols, n0, label_value=0, two_way_pairs=two_way_pairs
    )
    synth1 = sample_from_feature_model_with_pairs(
        rng, model1, num_cols, cat_cols, n1, label_value=1, two_way_pairs=two_way_pairs
    )

    synth_df = pd.concat([synth0, synth1], axis=0)
    synth_df = synth_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    return synth_df
def build_C1_global_dp_synth_label_conditional(
    train_df, num_cols, cat_cols, eps_total, rng, n_synth, cat_domains
):
    """
    Global label-conditional C1:
      1) DP class prior
      2) DP feature model for class 0
      3) DP feature model for class 1
      4) sample x|y
    """
    eps_total = float(eps_total)
    eps_y = 0.20 * eps_total
    eps_feat = 0.80 * eps_total

    py = dp_class_prior(rng, train_df, eps_y)

    df0 = train_df[train_df["target"] == 0].copy()
    df1 = train_df[train_df["target"] == 1].copy()

    def _public_fallback_model():
        return {
            "mu": {c: 0.5 for c in num_cols},
            "var": {c: 1.0 / 12.0 for c in num_cols},
            "cat_probs": {
                c: (
                    cat_domains[c],
                    np.ones(len(cat_domains[c]), dtype=float) / max(len(cat_domains[c]), 1)
                )
                for c in cat_cols
            }
        }

    model0 = dp_feature_model_fit(rng, df0, num_cols, cat_cols, eps_feat, cat_domains) if len(df0) >= 5 else _public_fallback_model()
    model1 = dp_feature_model_fit(rng, df1, num_cols, cat_cols, eps_feat, cat_domains) if len(df1) >= 5 else _public_fallback_model()

    n1 = int(round(n_synth * py))
    n1 = min(max(n1, 0), n_synth)
    n0 = n_synth - n1

    synth0 = sample_from_feature_model(rng, model0, num_cols, cat_cols, n0, label_value=0)
    synth1 = sample_from_feature_model(rng, model1, num_cols, cat_cols, n1, label_value=1)

    synth_df = pd.concat([synth0, synth1], axis=0)
    synth_df = synth_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    return synth_df

def build_C1_global_dp_synth(train_df, num_cols, cat_cols, eps_total, rng, n_synth, cat_domains):
    return dp_synth_sample(rng, dp_synth_fit(rng, train_df, num_cols, cat_cols, eps_total, cat_domains=cat_domains), num_cols, cat_cols, n_synth).reset_index(drop=True)

def strat_C1_global_synth(cfg, row_dict, synth_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(synth_df))
    shots = sample_balanced_shots(synth_df, k=k, seed=seed) if k > 0 else None
    if shots is not None:
      print("[DEBUG C1] shot label counts:",
      shots["target"].value_counts().to_dict())
    cols = num_cols + cat_cols
    system_msg = (
                "You are a robotic binary classifier.\n"
                "Output EXACTLY one character: 0 or 1.\n"
               "No punctuation. No explanation."
                  )
    user_msg = build_user_msg(cfg, row_dict, shots, cols)
    pred = llm_predict_binary(system_msg, user_msg)
    return pred

def build_C2a_clustered_dp_synth(train_df, num_cols, cat_cols, eps_cent, eps_mem, eps_synth, k, rng, rows_per_cluster, cluster_seed=42, min_cluster_size=5, clusters=None, cat_domains=None):
    n_features = len(num_cols) + len(cat_cols)
    eps_cent = float(eps_cent)
    eps_mem = float(eps_mem)
    eps_synth = float(eps_synth)

    e_cent_per_feature = eps_cent / max(n_features, 1)
    # Structure learning (non-DP): fixed clustering of clean train set with a fixed seed (trusted curator).
    if clusters is None:
        if len(num_cols) > 0:
            km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=cluster_seed).fit(train_df[num_cols])
        else:
            # Build feature matrix for clustering (safe for datasets with no cat or no num features)
            if len(cat_cols) > 0:
                X_cat = np.stack([train_df[c].astype('category').cat.codes.values for c in cat_cols], axis=1)
            else:
                X_cat = np.empty((len(train_df), 0), dtype=int)
            X_num = train_df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
            X_all = np.concatenate([X_num, X_cat], axis=1)
            if X_all.shape[1] == 0:
                raise ValueError('No features available for clustering: both num_cols and cat_cols are empty.')
            km = KMeans(n_clusters=k, n_init=1, max_iter=10, random_state=cluster_seed).fit(X_all)
        clusters = km.labels_
    if cat_domains is None:
        raise ValueError("cat_domains must be provided from the public schema for strict DP story.")

    dp_centroids = {}
    for c in np.unique(clusters):
        rows = train_df[clusters == c]
        dp_centroids[c] = {}

        for col in num_cols:
            e_sum = e_cent_per_feature / 2.0
            e_cnt = e_cent_per_feature / 2.0
            noisy_sum = float(rows[col].sum()) + rng.laplace(0, 1.0/e_sum)
            noisy_cnt = float(len(rows)) + rng.laplace(0, 1.0/e_cnt)
            dp_centroids[c][col] = noisy_sum / max(noisy_cnt, 1e-5)

        for col in cat_cols:
            dp_centroids[c][col] = dp_noisy_max(
                rng, Counter(rows[col].astype(str)),
                e_cent_per_feature,
                cat_domains[col]
            )

    distortion = compute_internal_distortion(train_df, clusters, dp_centroids, num_cols, cat_cols)

    cluster_ids = list(dp_centroids.keys())

    def _assign_then_rr(r):
        best = min(cluster_ids, key=lambda cid: mixed_dist(r.to_dict(), dp_centroids[cid], num_cols, cat_cols))
        return rr(rng, best, cluster_ids, eps_mem)

    mem = train_df.apply(_assign_then_rr, axis=1)

    synth_store = {}
    for c in cluster_ids:
        df_c = train_df[mem == c].copy()

        if len(df_c) < min_cluster_size:
            # Public fallback: keeps cluster stores defined without consuming privacy or reusing the full train.
            fallback = {
                "mu": {col: 0.5 for col in num_cols},
                "var": {col: 1.0/12.0 for col in num_cols},
                "cat_probs": {col: (cat_domains[col], (np.ones(len(cat_domains[col]))/max(len(cat_domains[col]),1))) for col in cat_cols},
                "py": 0.5
            }
            synth_store[c] = dp_synth_sample(rng, fallback, num_cols, cat_cols, rows_per_cluster).reset_index(drop=True)
        else:
            model = dp_synth_fit(rng, df_c, num_cols, cat_cols, eps_synth, cat_domains=cat_domains)
            synth_store[c] = dp_synth_sample(rng, model, num_cols, cat_cols, rows_per_cluster).reset_index(drop=True)

    return dp_centroids, synth_store, distortion

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
    shots = sample_balanced_shots(pool, k=k, seed=seed) if k > 0 else None

    if shots is not None:
      print(
        "[DEBUG C2a] seed:", seed,
        "| best cluster:", best_cluster,
        "| shot label counts:", shots["target"].value_counts().to_dict(),
        "| first rows:", shots.head(2).to_dict("records")
          )
    if shots is None or len(shots) == 0:
        return 0

    cols = num_cols + cat_cols
    system_msg = (
        "You are a robotic binary classifier.\n"
        "Output EXACTLY one character: 0 or 1.\n"
        "No punctuation. No explanation."
    )
    user_msg = build_user_msg(cfg, row_dict, shots, cols)
    pred = llm_predict_binary(system_msg, user_msg)
    return pred
import numpy as np
import pandas as pd

def run_epsilon_protocol(datasets_to_run=None, n_trials=50, k_list=None, results_csv="advisor_protocol_results.csv"):
    rows = []

    datasets = datasets_to_run if datasets_to_run is not None else DATASETS_TO_RUN
    trials = int(n_trials)
    k_values = k_list if k_list is not None else K_LIST
    trial_seeds = [1000 + i for i in range(trials)]
    for dataset_name in datasets:
        check_data_exists(dataset_name) # <--- The Safety Check!
        cfg = DATASET_META[dataset_name]
        print(f"\n=== DATASET: {dataset_name} ===")

        train_raw = pd.read_csv(cfg["path"])
        test_raw  = pd.read_csv(cfg["test"])

        train_df, num_cols, cat_cols = prep_dataset(train_raw, cfg, dataset_name)
        test_df,  _, _               = prep_dataset(test_raw,  cfg, dataset_name)

# Load public schema (domains + numeric bounds) and use fixed clipping/scaling.
        schema = load_dataset_schema_json(dataset_name)
        cat_domains, num_bounds = schema_to_domains_and_bounds(schema)
        cfg = dict(cfg)  # copy
        cfg["name"] = dataset_name
        cfg["cat_domains"] = cat_domains
        cfg["num_bounds"] = num_bounds

        # Override column typing using the public schema 
        feat_cols = [c for c in train_df.columns if c != "target"]
        num_cols = [c for c in feat_cols if c in num_bounds]
        cat_cols = [c for c in feat_cols if c in cat_domains and c not in num_cols]
        # Cast categorical columns to string for consistent RR / histograms
        for c in cat_cols:
            train_df[c] = train_df[c].astype(str)
            test_df[c]  = test_df[c].astype(str)

        if len(num_cols) > 0:
            train_df = clip_and_scale_numeric(train_df, num_cols, num_bounds)
            test_df  = clip_and_scale_numeric(test_df,  num_cols, num_bounds)

        EVAL_SEED = 777  # fixed seed for epsilon sweep
        eval_df = make_balanced_eval(test_df, per_class=EVAL_PER_CLASS, seed=EVAL_SEED)

        if eval_df is None or len(eval_df) == 0:
            continue

        y_true = eval_df["target"].astype(int).tolist()


        clean_pool = build_clean_pool(train_df, k_pool=min(5000, len(train_df)), seed=42)

        for K in k_values:
            for shots_used in SHOT_COUNTS:
                print(f"  K={K} | shots={shots_used}")

                # Build feature matrix for clustering (safe for datasets with no cat or no num features)
                if len(cat_cols) > 0:
                    X_cat = np.stack([train_df[c].astype('category').cat.codes.values for c in cat_cols], axis=1)
                else:
                    X_cat = np.empty((len(train_df), 0), dtype=int)
                X_num = train_df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
                X_all = np.concatenate([X_num, X_cat], axis=1)
                if X_all.shape[1] == 0:
                    raise ValueError('No features available for clustering: both num_cols and cat_cols are empty.')

                # Fixed (trusted-curator) clustering of clean training set
                # Fixed (trusted-curator) clustering of clean training set
                if len(num_cols) > 0:
                    clusters = KMeans(n_clusters=K, n_init=1, max_iter=10, random_state=CLUSTER_SEED).fit(train_df[num_cols]).labels_
                else:
                    # If schema defines no numeric features, cluster on categorical codes (trusted curator; structure only)
                    X = np.stack([train_df[c].astype("category").cat.codes.values for c in cat_cols], axis=1)
                    clusters = KMeans(n_clusters=K, n_init=1, max_iter=10, random_state=CLUSTER_SEED).fit(X_all).labels_

                C1_TWO_WAY_PAIRS = [
                    ("education", "occupation"),
                      ("marital-status", "relationship"),
                      ("workclass", "occupation"),
                      ]

                      # keep only pairs that exist in the current dataset
                C1_TWO_WAY_PAIRS = [
                      (a, b) for (a, b) in C1_TWO_WAY_PAIRS
                       if a in cat_cols and b in cat_cols and a in cat_domains and b in cat_domains
                        ]  
                C_TWO_WAY_PAIRS = [
                ("education", "occupation"),
                ("marital-status", "relationship"),
                ("workclass", "occupation"),
                                    ]

                C_TWO_WAY_PAIRS = [
                (a, b) for (a, b) in C_TWO_WAY_PAIRS
                if a in cat_cols and b in cat_cols and a in cat_domains and b in cat_domains
                                  ]   
                for trial_seed in trial_seeds:
                    rng_b1 = np.random.RandomState(trial_seed + 101)
                    rng_c1 = np.random.RandomState(trial_seed + 202)
                    rng_c2 = np.random.RandomState(trial_seed + 303)
                    b1_pool = build_B1_ldp_pool(train_df, num_cols, cat_cols, EPS_TOTAL, rng_b1, cat_domains)

                    c1_synth = build_C1_global_dp_synth_label_conditional_2way(
                        train_df,
                          num_cols,
                        cat_cols,
                      EPS_TOTAL,
                         rng_c1,
                        n_synth=len(train_df),
                       cat_domains=cat_domains,
                       two_way_pairs=C1_TWO_WAY_PAIRS,
                        )
                    c2_cent, c2_store, _ = build_C2a_clustered_dp_synth_label_conditional_2way(
                    train_df, num_cols, cat_cols,
                    EPS_C2A_CENT, EPS_C2A_MEM, EPS_C2A_SYN,
                    K, rng_c2, len(train_df) // max(K, 1),
                    clusters=clusters,
                    cat_domains=cat_domains,
                    two_way_pairs=C_TWO_WAY_PAIRS,
                                  )
                    for method in METHODS:
                      preds = []
                      for ridx, (_, row) in enumerate(eval_df.iterrows()):
                        row_dict = {col: row[col] for col in (num_cols + cat_cols)}

                        # keep trial fixed, but resample shots per query
                        row_seed = trial_seed * 10000 + ridx

                        if method == "A2":
                          pred = strat_A2_clean_fewshot(
                            cfg, row_dict, clean_pool, num_cols, cat_cols, shots_used, row_seed
                                                        )
                        elif method == "B1":
                          pred = strat_B1_ldp(
                            cfg, row_dict, b1_pool, num_cols, cat_cols, shots_used, row_seed
                                              )
                        elif method == "C1":
                          pred = strat_C1_global_synth(
                           cfg, row_dict, c1_synth, num_cols, cat_cols, shots_used, row_seed
                                                      )
                        elif method == "C2a":
                          pred = strat_C2a_clustered_synth(
                            cfg, row_dict, c2_cent, c2_store, num_cols, cat_cols, shots_used, row_seed
                                                          )

                        preds.append(int(pred))

                        # write immediately so disconnects don't lose progress
                      f1 = f1_score(y_true, preds)
                      acc = accuracy_score(y_true, preds)
                      row_out = {
                        "dataset": dataset_name,
                        "epsilon": EPS_TOTAL,
                        "K": K,
                        "shots": shots_used,
                        "seed": trial_seed,
                        "method": method,
                        "f1": f1,
                        "acc": acc,
                      }
                      append_row_to_csv(row_out, results_csv)
                      print(
                        f"[SAVED] {dataset_name} K={K} shots={shots_used} seed={trial_seed} method={method}",
                        flush=True
                        )

                cleanup()
                    
    print(f"\nSaved: {results_csv}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DP-GAN server runner")
    parser.add_argument("--mode", choices=["advisor", "run_all"], default="advisor")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--k_list", nargs="*", type=int, default=[8])
    parser.add_argument("--results_csv", default="advisor_protocol_results.csv")
    args = parser.parse_args()

    if args.mode == "run_all":
        df = run_all()
        try:
            df.to_csv("run_all_results.csv", index=False)
        except Exception:
            pass
    else:
        df = run_advisor_protocol(
            datasets_to_run=args.datasets,
            n_trials=args.trials,
            k_list=args.k_list,
            results_csv=args.results_csv,
        )
        try:
            df.to_csv(args.results_csv, index=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()
