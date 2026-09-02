#!/usr/bin/env python3
"""
Shared PriTabICL pipeline.

This module contains the common data preparation, privacy-preserving feature
weighting, weighted DP clustering, query routing, balanced demonstration
selection, Llama inference, metrics, and experiment orchestration used by the
PrivSyn and GEM backends.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from diffprivlib.models import KMeans as DPKMeans
    HAS_DIFFPRIVLIB_KMEANS = True
    DIFFPRIVLIB_IMPORT_ERROR = None
except Exception as exc:
    DPKMeans = None
    HAS_DIFFPRIVLIB_KMEANS = False
    DIFFPRIVLIB_IMPORT_ERROR = exc

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "datasets"

MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

PAPER_WEIGHT_SAMPLE_FRAC = 0.10
PAPER_CLUSTER_SHARE = 0.30
PAPER_DELTA = 1e-5
PAPER_K = 3
PAPER_SHOTS = 8
PAPER_WEIGHT_SPLIT_SEED = 314159
PAPER_WEIGHT_SUMMARY_BASE_SEED = 271828

WEIGHT_DP_SHOTS = 2
CAREY_SUPPORTED_K = [1, 2, 4, 8]
CAREY_NUMERIC_THRESHOLDS_ORIGINAL = {"Diabetes": {"Pregnancies": 4, "Age": 33}}

SERIALIZATION = "json"
DEBUG_PROMPT = False
DEBUG_PROMPT_ONCE = False
_prompt_debug_printed = False

tokenizer = None
model = None
qwen_tokenizer = None
qwen_model = None

DATASET_META = {
    "Adult": {
        "path": str(DATA_ROOT / "adult" / "train.csv"),
        "test": str(DATA_ROOT / "adult" / "test.csv"),
        "schema_file": "adult.json",
        "label": "label",
        "pos": ">50K",
        "task": "income prediction",
        "decode_prompt_values": True,
    },
    "Airline": {
        "path": str(DATA_ROOT / "airline" / "train.csv"),
        "test": str(DATA_ROOT / "airline" / "test.csv"),
        "schema_file": "airline.json",
        "label": "satisfaction",
        "pos": "satisfied",
        "task": "passenger satisfaction prediction",
        "decode_prompt_values": True,
    },
    "Diabetes": {
        "path": str(DATA_ROOT / "diabetes" / "train.csv"),
        "test": str(DATA_ROOT / "diabetes" / "test.csv"),
        "schema_file": "diabetes.json",
        "label": "Diabetes_binary",
        "pos": "1",
        "task": "diabetes risk prediction",
        "decode_prompt_values": False,
    },
    "Phishing": {
        "path": str(DATA_ROOT / "phishing" / "train.csv"),
        "test": str(DATA_ROOT / "phishing" / "test.csv"),
        "schema_file": "phishing.json",
        "label": "label",
        "pos": "1",
        "task": "phishing website detection",
        "decode_prompt_values": False,
    },
}


def load_inference_model():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return tokenizer, model

    print(f"Loading inference model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model


def load_weighting_model():
    global qwen_tokenizer, qwen_model
    if qwen_tokenizer is not None and qwen_model is not None:
        return qwen_tokenizer, qwen_model

    print(f"Loading weighting model: {QWEN_MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, use_fast=True)
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    qwen_model.config.pad_token_id = qwen_tokenizer.pad_token_id
    qwen_model.eval()
    return qwen_tokenizer, qwen_model


def load_dataset_schema_json(dataset_name, schema_dir=None):
    cfg = DATASET_META[dataset_name]
    if schema_dir is None:
        schema_path = Path(cfg["path"]).parent / cfg["schema_file"]
    else:
        schema_path = Path(schema_dir) / cfg["schema_file"]
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_data_exists(dataset_name):
    if dataset_name not in DATASET_META:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    cfg = DATASET_META[dataset_name]
    missing = [p for p in (cfg["path"], cfg["test"]) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Dataset files not found for {dataset_name}: {missing}. "
            f"Expected them under {DATA_ROOT}."
        )


def append_row_to_csv(row: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df_row = pd.DataFrame([row])
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    df_row.to_csv(csv_path, mode='a', header=not exists, index=False)


def schema_to_domains_and_bounds(schema):
    cat_domains = {}
    num_bounds = {}
    if not schema:
        return (cat_domains, num_bounds)
    for col in schema.get('columns', []):
        name = col.get('name')
        ctype = col.get('type')
        if ctype == 'categorical':
            if 'i2s' in col and isinstance(col['i2s'], list):
                cat_domains[name] = list(map(str, col['i2s']))
        elif ctype == 'continuous':
            if 'min' in col and 'max' in col:
                num_bounds[name] = (float(col['min']), float(col['max']))
    return (cat_domains, num_bounds)


def clip_and_scale_numeric(df, num_cols, num_bounds):
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


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_laplace(rng, scale):
    if scale <= 0 or np.isinf(scale) or np.isnan(scale):
        return 0.0
    return float(rng.laplace(0.0, scale))


def _is_infinite_epsilon(eps):
    return eps is None or np.isinf(eps) or float(eps) > 1e+20


def _scaled_threshold_from_original(col, threshold_original, num_bounds):
    if col not in num_bounds:
        raise ValueError(f'Missing public numeric bounds for Carey threshold column: {col}')
    L, U = num_bounds[col]
    if U <= L:
        raise ValueError(f'Invalid numeric bounds for {col}: {(L, U)}')
    return float((float(threshold_original) - float(L)) / (float(U) - float(L)))


def _carey_explicit_group_value(dataset_name, col, value):
    """Exact semantic GROUP BY mappings reported by DP-TabICL where available."""
    ds = str(dataset_name).lower()
    c = str(col).lower()
    v = str(value).strip().lower()
    if ds == 'adult' and c == 'race':
        return 'White' if v == 'white' else 'Non-White'
    if ds in {'bank', 'banking'} and c in {'marital', 'marital-status'}:
        return 'Married' if v == 'married' else 'Not-Married'
    if ds in {'bank', 'banking'} and c in {'contact', 'contact-type'}:
        return 'Cellular' if v == 'cellular' else 'Non-Cellular'
    return None


def _public_binary_group_series(df, col, num_cols, cat_cols, cat_domains, num_bounds, dataset_name=None):
    dataset_name = str(dataset_name) if dataset_name is not None else ''
    if col == 'target':
        return df[col].astype(int).astype(str)
    if col in num_cols:
        threshold_scaled = 0.5
        if dataset_name in CAREY_NUMERIC_THRESHOLDS_ORIGINAL and col in CAREY_NUMERIC_THRESHOLDS_ORIGINAL[dataset_name]:
            threshold_original = CAREY_NUMERIC_THRESHOLDS_ORIGINAL[dataset_name][col]
            threshold_scaled = _scaled_threshold_from_original(col, threshold_original, num_bounds)
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(0.0, 1.0)
        return pd.Series(np.where(vals.to_numpy() <= threshold_scaled, 'low', 'high'), index=df.index)
    if col in cat_cols:
        if col not in cat_domains:
            raise ValueError(f'Missing public categorical domain for GROUP BY column: {col}')
        domain = list(map(str, cat_domains[col]))
        vals = df[col].astype(str)
        mapped = vals.map(lambda x: _carey_explicit_group_value(dataset_name, col, x))
        if mapped.notna().all():
            return mapped.astype(str)
        if len(domain) <= 2:
            return vals
        midpoint = int(np.ceil(len(domain) / 2))
        group0 = set(domain[:midpoint])
        return vals.apply(lambda x: 'group0' if str(x) in group0 else 'group1')
    raise ValueError(f'Unknown GROUP BY column: {col}')


def _public_binary_group_domain(col, num_cols, cat_cols, cat_domains, dataset_name=None):
    if col == 'target':
        return ['0', '1']
    if col in num_cols:
        return ['low', 'high']
    if col in cat_cols:
        if col not in cat_domains:
            raise ValueError(f'Missing public categorical domain for GROUP BY column: {col}')
        ds = str(dataset_name).lower() if dataset_name is not None else ''
        c = str(col).lower()
        if ds == 'adult' and c == 'race':
            return ['White', 'Non-White']
        if ds in {'bank', 'banking'} and c in {'marital', 'marital-status'}:
            return ['Married', 'Not-Married']
        if ds in {'bank', 'banking'} and c in {'contact', 'contact-type'}:
            return ['Cellular', 'Non-Cellular']
        domain = list(map(str, cat_domains[col]))
        if len(domain) <= 2:
            return domain
        return ['group0', 'group1']
    raise ValueError(f'Unknown GROUP BY column: {col}')


def _first_existing(preferred, available):
    available = set(available)
    for c in preferred:
        if c in available:
            return c
    return None


def gdp_groupby_spec_for_dataset(dataset_name, k_shots, num_cols, cat_cols):
    dataset_name = str(dataset_name)
    if k_shots not in CAREY_SUPPORTED_K:
        k_base = 8
    else:
        k_base = int(k_shots)
    if k_base == 1:
        return []
    if k_base == 2:
        return ['target']
    available = set(num_cols) | set(cat_cols)
    if dataset_name == 'Adult':
        f1 = _first_existing(['sex', 'gender'], available)
        f2 = _first_existing(['race', 'marital-status', 'relationship', 'education'], available)
    elif dataset_name in {'Bank', 'Banking'}:
        f1 = _first_existing(['marital', 'marital-status'], available)
        f2 = _first_existing(['contact', 'contact-type'], available)
    elif dataset_name == 'Diabetes':
        f1 = _first_existing(['Pregnancies'], available)
        f2 = _first_existing(['Age'], available)
    elif dataset_name == 'Abalone':
        f1 = _first_existing(['sex'], available)
        f2 = _first_existing(list(num_cols), available)
    elif dataset_name == 'Shoppers':
        f1 = _first_existing(['VisitorType', 'Weekend', 'Month'], available)
        f2 = _first_existing(['Weekend', 'Month', 'VisitorType'], available)
        if f2 == f1:
            f2 = _first_existing([c for c in list(cat_cols) + list(num_cols) if c != f1], available)
    elif dataset_name == 'DefaultCredit':
        f1 = _first_existing(list(cat_cols), available)
        f2 = _first_existing([c for c in list(cat_cols) + list(num_cols) if c != f1], available)
    else:
        f1 = _first_existing(list(cat_cols) + list(num_cols), available)
        f2 = _first_existing([c for c in list(cat_cols) + list(num_cols) if c != f1], available)
    if k_base == 4:
        return ['target'] + ([f1] if f1 is not None else [])
    if k_base == 8:
        out = ['target']
        if f1 is not None:
            out.append(f1)
        if f2 is not None:
            out.append(f2)
        return out
    raise ValueError(f'Unsupported k_base: {k_base}')


def gdp_noisy_categorical_argmax(rng, values, domain, epsilon_col):
    domain = list(map(str, domain))
    if len(domain) == 0:
        return None
    vals = pd.Series(values).astype(str) if values is not None else pd.Series([], dtype=str)
    if _is_infinite_epsilon(epsilon_col):
        if len(vals) == 0:
            return domain[0]
        counts = Counter(vals.tolist())
        return max(domain, key=lambda x: counts.get(str(x), 0))
    epsilon_col = float(epsilon_col)
    if epsilon_col <= 0:
        raise ValueError('epsilon_col must be positive')
    counts = Counter(vals.tolist())
    noisy_counts = {}
    for v in domain:
        noisy_counts[str(v)] = counts.get(str(v), 0) + _rng_laplace(rng, scale=1.0 / epsilon_col)
    return max(noisy_counts, key=noisy_counts.get)


def gdp_noisy_numeric_average_scaled(rng, values, epsilon_col, min_count=1.0):
    vals = pd.to_numeric(pd.Series(values), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    if len(vals) == 0 and _is_infinite_epsilon(epsilon_col):
        return 0.5
    if _is_infinite_epsilon(epsilon_col):
        if len(vals) == 0:
            return 0.5
        return float(np.clip(vals.mean(), 0.0, 1.0))
    epsilon_col = float(epsilon_col)
    if epsilon_col <= 0:
        raise ValueError('epsilon_col must be positive')
    eps_count = epsilon_col / 2.0
    eps_sum = epsilon_col / 2.0
    true_count = float(len(vals))
    true_sum = float(vals.sum())
    noisy_count = true_count + _rng_laplace(rng, scale=1.0 / eps_count)
    noisy_sum = true_sum + _rng_laplace(rng, scale=1.0 / eps_sum)
    noisy_count = max(float(noisy_count), float(min_count))
    avg = noisy_sum / noisy_count
    return float(np.clip(avg, 0.0, 1.0))


def build_GDP_tabicl_demos(train_df, num_cols, cat_cols, eps_total, k_shots, rng, cat_domains, dataset_name=None, num_bounds=None, sample_rate=1.0, groupby_spec=None, min_noisy_count=1.0):
    if cat_domains is None:
        raise ValueError('cat_domains must be provided from public schema.')
    if num_bounds is None:
        num_bounds = {}
    dataset_name = str(dataset_name) if dataset_name is not None else ''
    df = train_df.copy().reset_index(drop=True)
    for c in cat_cols:
        df[c] = df[c].astype(str)
    df['target'] = df['target'].astype(int)
    if sample_rate < 1.0:
        if sample_rate <= 0:
            raise ValueError('sample_rate must be in (0,1].')
        mask = rng.rand(len(df)) < float(sample_rate)
        sampled = df.loc[mask].copy().reset_index(drop=True)
        if len(sampled) == 0:
            sampled = df.sample(n=1, random_state=int(rng.randint(0, 10 ** 9))).copy().reset_index(drop=True)
    else:
        sampled = df.copy().reset_index(drop=True)
    if groupby_spec is None:
        groupby_spec = gdp_groupby_spec_for_dataset(dataset_name=dataset_name, k_shots=k_shots, num_cols=num_cols, cat_cols=cat_cols)
    groupby_spec = [c for c in groupby_spec if c is not None]
    if k_shots not in CAREY_SUPPORTED_K:
        print(f'[WARN GDP] k_shots={k_shots} is not in {CAREY_SUPPORTED_K}. Using Carey-style grouping and then post-processing to requested k.')
    working = sampled.copy()
    key_cols = []
    key_domains = []
    if len(groupby_spec) == 0:
        working['__gdp_key_all'] = '__ALL__'
        key_cols = ['__gdp_key_all']
        key_domains = [['__ALL__']]
    else:
        for col in groupby_spec:
            key_col = f'__gdp_key_{col}'
            working[key_col] = _public_binary_group_series(working, col, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, num_bounds=num_bounds, dataset_name=dataset_name).astype(str)
            key_cols.append(key_col)
            key_domains.append(_public_binary_group_domain(col, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, dataset_name=dataset_name))
    public_keys = list(product(*key_domains))
    private_cols = list(num_cols) + list(cat_cols) + ['target']
    if _is_infinite_epsilon(eps_total):
        epsilon_col = np.inf
    else:
        epsilon_col = float(eps_total) / max(len(private_cols), 1)
    demos = []
    for key_tuple in public_keys:
        mask = np.ones(len(working), dtype=bool)
        for key_col, key_val in zip(key_cols, key_tuple):
            mask &= working[key_col].astype(str).to_numpy() == str(key_val)
        group = working.loc[mask].copy()
        demo = {}
        for col in num_cols:
            demo[col] = gdp_noisy_numeric_average_scaled(rng=rng, values=group[col] if len(group) > 0 else pd.Series([], dtype=float), epsilon_col=epsilon_col, min_count=min_noisy_count)
        for col in cat_cols:
            if col not in cat_domains:
                raise ValueError(f'Missing categorical domain for GDP column: {col}')
            demo[col] = gdp_noisy_categorical_argmax(rng=rng, values=group[col] if len(group) > 0 else pd.Series([], dtype=str), domain=cat_domains[col], epsilon_col=epsilon_col)
        label_val = gdp_noisy_categorical_argmax(rng=rng, values=group['target'].astype(str) if len(group) > 0 else pd.Series([], dtype=str), domain=['0', '1'], epsilon_col=epsilon_col)
        demo['target'] = int(label_val)
        demos.append(demo)
    demos_df = pd.DataFrame(demos)
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    demos_df = demos_df[ordered_cols].copy()
    if len(demos_df) > k_shots:
        demos_df = demos_df.sample(n=int(k_shots), replace=False, random_state=int(rng.randint(0, 10 ** 9))).reset_index(drop=True)
    elif len(demos_df) < k_shots:
        demos_df = demos_df.sample(n=int(k_shots), replace=True, random_state=int(rng.randint(0, 10 ** 9))).reset_index(drop=True)
    print('[DEBUG GDP]', 'dataset:', dataset_name, '| eps:', eps_total, '| k:', k_shots, '| sample_rate:', sample_rate, '| groupby:', groupby_spec, '| demos:', len(demos_df), '| labels:', demos_df['target'].value_counts().to_dict())
    return demos_df.reset_index(drop=True)


def cleanup():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import jax
        jax.clear_caches()
    except ImportError:
        pass
    except AttributeError:
        try:
            jax.clear_backends()
        except Exception:
            pass


def clip_numeric_df(df, num_cols, lo=0.0, hi=1.0):
    df = df.copy()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(lo).clip(lo, hi)
    return df


def _fit_equal_width_bin_edges(df, num_cols, n_bins=16, lo=0.0, hi=1.0):
    edges = {}
    for c in num_cols:
        edges[c] = np.linspace(lo, hi, n_bins + 1)
    return edges


def prep_dataset(df, cfg, name):
    df = df.copy().replace('?', np.nan).dropna()

    if name == 'Abalone':
        df.columns = [c.strip().lower() for c in df.columns]
        cfg = dict(cfg)
        cfg['label'] = cfg['label'].strip().lower()

    label_col = cfg['label']

    label_numeric = pd.to_numeric(df[label_col], errors='coerce')
    positive_numeric = pd.to_numeric(
        pd.Series([cfg['pos']]),
        errors='coerce'
    ).iloc[0]

    if label_numeric.notna().all() and pd.notna(positive_numeric):
        df['target'] = (
            label_numeric == float(positive_numeric)
        ).astype(int)
    else:
        df['target'] = (
            df[label_col].astype(str).str.strip().str.upper()
            == str(cfg['pos']).strip().upper()
        ).astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in ['target', label_col]:
        if col in num_cols:
            num_cols.remove(col)

    cat_cols = [
        col for col in df.columns
        if col not in num_cols and col not in ['target', label_col]
    ]

    return (
        df[num_cols + cat_cols + ['target']].dropna(),
        num_cols,
        cat_cols
    )


def make_balanced_eval(test_df, per_class, seed):
    pos = test_df[test_df['target'] == 1]
    neg = test_df[test_df['target'] == 0]
    n = min(len(pos), len(neg), per_class)
    if n == 0:
        return None
    return pd.concat([pos.sample(n, random_state=seed), neg.sample(n, random_state=seed)]).sample(frac=1, random_state=seed).reset_index(drop=True)


def format_value(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'NA'
    except Exception:
        pass
    if isinstance(v, (float, np.floating)):
        return f'{float(v):.4f}'
    s = str(v)
    return s[:60]


def _carey_get(row_dict, col, default='NA'):
    return format_value(row_dict.get(col, default))


def _carey_yesno_label(v):
    try:
        return 'Yes' if int(v) == 1 else 'No'
    except Exception:
        s = str(v).strip().lower()
        return 'Yes' if s in ['1', 'yes', 'true', 'positive'] else 'No'


def serialize_row_carey_style(cfg: dict, row_dict: dict, cols: list, include_label: bool=True) -> str:
    dataset = str(cfg.get('name', '')).lower()

    def label_suffix(question):
        if include_label:
            return f"{question} Yes or No? Answer: {_carey_yesno_label(row_dict.get('target', 0))}"
        return f'{question} Yes or No? Answer:'
    if dataset == 'adult':
        text = f"An individual recorded in the 1994 US census is described as follows: This person is {_carey_get(row_dict, 'age')} years old. Their workclass is {_carey_get(row_dict, 'workclass')}. Their education is {_carey_get(row_dict, 'education')}. Their marital status is {_carey_get(row_dict, 'marital-status')}. Their occupation is {_carey_get(row_dict, 'occupation')}. Their relationship status is {_carey_get(row_dict, 'relationship')}. Their race is {_carey_get(row_dict, 'race')}. Their sex is {_carey_get(row_dict, 'sex')}. Their capital gain is {_carey_get(row_dict, 'capital-gain')}. Their capital loss is {_carey_get(row_dict, 'capital-loss')}. They work {_carey_get(row_dict, 'hours-per-week')} hours per week. Their native country is {_carey_get(row_dict, 'native-country')}. "
        return text + label_suffix('Does this person earn more than 50,000 dollars annually?')
    if dataset == 'banking':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'A client contacted during a direct marketing campaign by a banking institution is described as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Will this client subscribe to a term deposit?')
    if dataset == 'diabetes':
        text = f"The following describes diagnostic measurements for a patient. The patient has been pregnant {_carey_get(row_dict, 'Pregnancies')} times. The plasma glucose concentration is {_carey_get(row_dict, 'Glucose')}. The blood pressure is {_carey_get(row_dict, 'BloodPressure')}. The skin thickness is {_carey_get(row_dict, 'SkinThickness')}. The insulin value is {_carey_get(row_dict, 'Insulin')}. The body mass index is {_carey_get(row_dict, 'BMI')}. The diabetes pedigree function is {_carey_get(row_dict, 'DiabetesPedigreeFunction')}. The age is {_carey_get(row_dict, 'Age')}. "
        return text + label_suffix('Does this patient have diabetes?')
    if dataset == 'magic':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'A particle event recorded by an atmospheric Cherenkov gamma telescope is described as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Is this event a gamma signal?')
    if dataset == 'phishing':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'A website is described using URL and webpage attributes as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Is this website a phishing website?')
    if dataset == 'abalone':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'An abalone specimen is described as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Is this abalone in the positive age class?')
    if dataset == 'shoppers':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'An online shopping session is described as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Does this session lead to a purchase?')
    if dataset == 'defaultcredit':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'A credit card client is described using demographic, payment, bill, and repayment attributes as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Will this client default on payment?')
    if dataset == 'airline':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'An airline passenger is described as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Is this passenger satisfied with their overall flight experience?')

    if dataset == 'weather':
        feature_text = []
        for c in cols:
            if c == 'target':
                continue
            feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
        text = 'The weather observations at a specific monitoring station are described as follows: ' + '; '.join(feature_text) + '. '
        return text + label_suffix('Will it rain tomorrow?')
    feature_text = []
    for c in cols:
        if c == 'target':
            continue
        feature_text.append(f"{str(c).replace('_', ' ')} is {_carey_get(row_dict, c)}")
    text = f"An instance from the {cfg.get('task', 'binary classification')} dataset is described as follows: " + '; '.join(feature_text) + '. '
    return text + label_suffix('Is the correct label positive?')


def serialize_row_to_text_dpticl(row_dict: dict, cols: list) -> str:

    def clean_name(c):
        return str(c).replace('-', ' ').replace('_', ' ')

    def clean_value(v):
        return format_value(v).replace('-', ' ')
    parts = ['An individual recorded in the 1994 US census has the following attributes:']
    for c in cols:
        if c == 'target':
            continue
        val = clean_value(row_dict.get(c))
        parts.append(f'{clean_name(c)} is {val}')
    return '; '.join(parts) + '.'


def build_prompt_metadata(schema):
    """
    Read optional LLM-facing metadata from the public schema.

    Internal CSV values and DP domains remain unchanged.
    """
    prompt_names = {}
    value_labels = {}
    units = {}

    if not isinstance(schema, dict):
        return prompt_names, value_labels, units

    for col in schema.get('columns', []):
        name = col.get('name')

        if name is None:
            continue

        name = str(name)

        # Optional readable feature name shown only to the inference LLM.
        prompt_names[name] = str(
            col.get('prompt_name', name)
        )

        # Optional mapping from raw category code to readable text.
        labels = col.get('value_labels')
        if isinstance(labels, dict):
            value_labels[name] = {
                str(raw): str(readable)
                for raw, readable in labels.items()
            }

        # Optional unit appended to numerical values.
        unit = col.get('unit')
        if unit is not None:
            units[name] = str(unit)

    return prompt_names, value_labels, units


def serialize_row_json(
    cfg: dict,
    row_dict: dict,
    cols: list,
    include_label: bool = True,
) -> str:
    """
    Serialize a row for Llama inference.

    Old datasets retain the exact legacy representation unless
    decode_prompt_values=True is set in DATASET_META.
    """
    obj = {}

    decode_values = bool(
        cfg.get('decode_prompt_values', False)
    )

    schema = cfg.get('schema')
    prompt_names, value_labels, units = build_prompt_metadata(schema)

    for c in cols:
        if c == 'target':
            continue

        raw_value = format_value(row_dict.get(c))

        if decode_values:
            shown_name = prompt_names.get(c, c)

            shown_value = value_labels.get(c, {}).get(
                str(raw_value),
                raw_value,
            )

            if c in units:
                shown_value = f'{shown_value} {units[c]}'
        else:
            # Exact legacy behavior used by completed datasets.
            shown_name = c
            shown_value = raw_value

        obj[shown_name] = shown_value

    if include_label:
        obj['label'] = str(int(row_dict.get('target', 0)))

    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(',', ':'),
    )


def build_user_msg(cfg: dict, row_dict: dict, shots_df, cols: list) -> str:
    global SERIALIZATION
    task_name = cfg.get('task', 'binary classification')
    header = [f'Use the labeled examples to perform {task_name} for the query instance.', 'Allowed labels: 0 or 1.', 'Return only 0 or 1.', '']
    if SERIALIZATION == 'natural':
        parts = list(header)
        if shots_df is not None and len(shots_df) > 0:
            for idx, (_, s) in enumerate(shots_df.iterrows()):
                text_repr = serialize_row_to_text_dpticl(s.to_dict(), cols)
                label_str = str(int(s['target']))
                parts.append(f'Example {idx + 1}: {text_repr} Label: {label_str}')
        target_repr = serialize_row_to_text_dpticl(row_dict, cols)
        parts.append(f'Query: {target_repr} Label:')
        return '\n'.join(parts)
    elif SERIALIZATION == 'json':
        parts = list(header)
    
        if shots_df is not None and len(shots_df) > 0:
            for idx, (_, s) in enumerate(shots_df.iterrows()):
                label_str = str(int(s['target']))
    
                serialized_example = serialize_row_json(
                    cfg,
                    s.to_dict(),
                    cols,
                    include_label=False,
                )
    
                parts.append(f'Example {idx + 1}:')
                parts.append(f'Input: {serialized_example}')
                parts.append(f'Label: {label_str}')
                parts.append('')
    
        serialized_query = serialize_row_json(
            cfg,
            row_dict,
            cols,
            include_label=False,
        )
    
        parts.append('Query:')
        parts.append(f'Input: {serialized_query}')
        parts.append('Label:')
    
        return '\n'.join(parts)
    elif SERIALIZATION == 'carey':
        parts = [f'Use the labeled examples to perform {task_name}.', 'Answer the final question with only Yes or No.', '']
        if shots_df is not None and len(shots_df) > 0:
            for idx, (_, s) in enumerate(shots_df.iterrows()):
                parts.append(f'Example {idx + 1}:')
                parts.append(serialize_row_carey_style(cfg, s.to_dict(), cols, include_label=True))
                parts.append('')
        parts.append('Query:')
        parts.append(serialize_row_carey_style(cfg, row_dict, cols, include_label=False))
        return '\n'.join(parts)
    else:
        raise ValueError(f'Unknown SERIALIZATION: {SERIALIZATION}')


def execute_classification_strategy(cfg, row_dict, shots, cols):
    num_bounds = cfg.get('num_bounds', {})
    clean_row = unscale_row_dict(row_dict, num_bounds)
    if shots is not None and len(shots) > 0:
        clean_shots = shots.copy()
        for c, (L, U) in num_bounds.items():
            if c in clean_shots.columns:
                clean_shots[c] = pd.to_numeric(clean_shots[c], errors='coerce') * (U - L) + L
                clean_shots[c] = clean_shots[c].round(2)
    else:
        clean_shots = shots
    if SERIALIZATION == 'carey':
        system_msg = 'Follow the labeled examples. Output exactly one answer: Yes or No.'
    else:
        system_msg = 'Follow the labeled examples. Output exactly one label: 0 or 1.'
    user_msg = build_user_msg(cfg, clean_row, clean_shots, cols)
    global _prompt_debug_printed
    if DEBUG_PROMPT and (not DEBUG_PROMPT_ONCE or not _prompt_debug_printed):
        print('\n' + '=' * 80)
        print('[DEBUG] SYSTEM MESSAGE')
        print(system_msg)
        print('-' * 80)
        print('[DEBUG] USER MESSAGE')
        print(user_msg)
        print('=' * 80 + '\n')
    return llm_predict_binary(system_msg, user_msg)


def llm_predict_binary(system_msg: str, user_msg: str) -> int:
    load_inference_model()
    prompt_txt = system_msg.strip() + '\n\n' + user_msg.rstrip()
    if SERIALIZATION == 'carey':
        if not prompt_txt.endswith('Answer:'):
            prompt_txt += '\nAnswer:'
    elif not prompt_txt.endswith('Label:'):
        prompt_txt += '\nLabel:'
    prompt_txt += ' '
    enc = tokenizer(prompt_txt, return_tensors='pt', truncation=True, max_length=4000, add_special_tokens=True).to(model.device)
    outputs = model(**enc)
    next_token_logits = outputs.logits[0, -1, :]

    def get_max_score(token_strings):
        scores = []
        for s in token_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 0:
                continue
            token_id = ids[-1]
            scores.append(next_token_logits[token_id].item())
        if len(scores) == 0:
            return -1e+30
        return max(scores)
    if SERIALIZATION == 'carey':
        score_0 = get_max_score(['No', ' No', 'no', ' no'])
        score_1 = get_max_score(['Yes', ' Yes', 'yes', ' yes'])
    else:
        score_0 = get_max_score(['0', ' 0'])
        score_1 = get_max_score(['1', ' 1'])
    pred = 1 if score_1 > score_0 else 0
    if not hasattr(llm_predict_binary, "_debug_count"):
        llm_predict_binary._debug_count = 0
    
    if llm_predict_binary._debug_count < 5:
        print(
            "\n[INFERENCE DEBUG]",
            "\nscore_0:", score_0,
            "\nscore_1:", score_1,
            "\nprediction:", int(score_1 > score_0),
            "\nprompt:\n", prompt_txt,
            flush=True,
        )
    
        llm_predict_binary._debug_count += 1
    del enc, outputs, next_token_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def sample_balanced_shots(pool_df: pd.DataFrame, k: int, seed: int, label_col: str='target'):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None
    k = int(min(k, len(pool_df)))
    if k <= 0:
        return None
    pos = pool_df[pool_df[label_col] == 1]
    neg = pool_df[pool_df[label_col] == 0]
    if len(pos) == 0 or len(neg) == 0:
        shots = pool_df.sample(n=k, replace=k > len(pool_df), random_state=seed)
        return shots.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    k_pos = k // 2
    k_neg = k - k_pos
    s_pos = pos.sample(n=k_pos, replace=k_pos > len(pos), random_state=seed)
    s_neg = neg.sample(n=k_neg, replace=k_neg > len(neg), random_state=seed + 1)
    shots = pd.concat([s_pos, s_neg], axis=0)
    shots = shots.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    print('[DEBUG shots] seed =', seed, '| labels =', shots[label_col].tolist())
    return shots


def unscale_row_dict(row_dict, num_bounds):
    out = dict(row_dict)
    for c, (L, U) in num_bounds.items():
        if c in out:
            try:
                val = float(out[c]) * (U - L) + L
                out[c] = round(val, 2)
            except Exception:
                pass
    return out


def _public_target_description(cfg, schema):
    task = str(cfg.get('task', 'binary classification')).strip()
    label_name = None
    label_values = []
    if isinstance(schema, dict):
        label_name = schema.get('label')
        for col in schema.get('columns', []):
            if col.get('name') in {'label', label_name, cfg.get('label')} and col.get('type') == 'categorical':
                label_values = list(map(str, col.get('i2s', [])))
                if label_values:
                    break
    if label_values:
        return f"{task}; target classes are {label_values}"
    return task


def _public_feature_description(feature, schema, num_cols, cat_cols):
    meta = None
    if isinstance(schema, dict):
        for col in schema.get('columns', []):
            if str(col.get('name')) == str(feature):
                meta = col
                break
    
    ftype = 'continuous' if feature in num_cols else 'categorical' if feature in cat_cols else 'unknown'
    
    desc_parts = [f"Feature name: {feature}", f"Type: {ftype}"]
    
    if meta:
        if 'description' in meta:
            desc_parts.append(f"Description: {meta['description']}")
        if 'min' in meta and 'max' in meta:
            unit = meta.get('unit', '')
            desc_parts.append(f"Real-World Range: [{meta['min']} to {meta['max']}] {unit}".strip())
        if 'value_labels' in meta:
            desc_parts.append(f"Value Meanings: {json.dumps(meta['value_labels'])}")
            
    return "\n".join(desc_parts)


def _extract_json_object(text):
    text = str(text).strip()
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r'([\{,]\s*)[/]?([a-zA-Z0-9_ -]+)":', r'\1"\2":', text)
    candidates = [text]
    first = text.find('{')
    last = text.rfind('}')
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def split_disjoint_weight_sample(
    train_df,
    sample_frac,
    seed,
    label_col="target",
):
    """Uniform, data-independent, disjoint prototype/main split."""
    if not 0.0 < float(sample_frac) < 1.0:
        raise ValueError("sample_frac must be strictly between 0 and 1.")

    n = len(train_df)
    if n < 2:
        raise ValueError("Need at least two records for a disjoint split.")

    n_weight = int(np.floor(float(sample_frac) * n))
    n_weight = max(1, min(n_weight, n - 1))

    rng = np.random.RandomState(int(seed))
    all_indices = train_df.index.to_numpy()
    chosen = rng.choice(all_indices, size=n_weight, replace=False)

    chosen_set = set(chosen.tolist())
    main_indices = [idx for idx in all_indices if idx not in chosen_set]

    weight_df = train_df.loc[chosen].copy().reset_index(drop=True)
    main_df = train_df.loc[main_indices].copy().reset_index(drop=True)

    if len(weight_df) + len(main_df) != n:
        raise AssertionError("Partition does not cover the dataset.")
    if len(set(chosen).intersection(main_indices)) != 0:
        raise AssertionError("Prototype and main partitions overlap.")

    return weight_df, main_df


def build_dp_weighting_demos(weight_df, num_cols, cat_cols, epsilon, rng,
                             cat_domains, dataset_name, num_bounds):
    """Create two DP class-conditioned prototypes from the reserved partition.

    This reuses the existing GDP-style noisy averages/modes. The reserved rows
    consume the full epsilon for this branch and are excluded from the main branch.
    """
    return build_GDP_tabicl_demos(
        train_df=weight_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        eps_total=float(epsilon),
        k_shots=WEIGHT_DP_SHOTS,
        rng=rng,
        cat_domains=cat_domains,
        dataset_name=dataset_name,
        num_bounds=num_bounds,
        sample_rate=1.0,
        groupby_spec=['target'],
    )


def build_llm_weights_from_dp_demos(cfg, dp_demos, num_cols, cat_cols, schema=None):
    load_weighting_model()
    all_cols = list(num_cols) + list(cat_cols)
    if not all_cols:
        return {}
        
    num_bounds = cfg.get('num_bounds', {})
    task_description = _public_target_description(cfg, schema)
    
    demo_lines = []
    for i, (_, row) in enumerate(dp_demos.iterrows(), start=1):
        unscaled_vals = []
        for c in all_cols:
            val = row[c]
            # Unscale numeric values for prompt readability
            if c in num_bounds:
                L, U = num_bounds[c]
                real_val = float(val) * (U - L) + L
                unscaled_vals.append(f"{c}={real_val:.2f}")
            else:
                unscaled_vals.append(f"{c}={format_value(val)}")
                
        demo_lines.append(f"Class summary (label={int(row['target'])}): " + ", ".join(unscaled_vals))

    feature_lines = "\n\n".join(
        _public_feature_description(c, schema, num_cols, cat_cols) for c in all_cols
    )

    system_msg = (
        "You are a clinical data scientist evaluating feature importance for tabular classification.\n"
        "Calculate the mean shift (delta) between label=0 and label=1 for each feature, "
        "and combine it with clinical domain relevance.\n"
        "Assign continuous weights between 0.05 (irrelevant) and 1.00 (critical predictor).\n"
        "Avoid assigning default flat weights (like 0.5) to multiple features unless they have identical effect sizes."
    )

    user_msg = (
        f"Prediction Task: {task_description}\n\n"
        "Class-Conditioned Privacy-Preserving Summaries:\n"
        + "\n".join(demo_lines) + "\n\n"
        "Feature Metadata:\n"
        + feature_lines + "\n\n"
        "Return ONLY a valid JSON dictionary mapping every feature to a score in [0.05, 1.00]."
    )
    
    
    # Use Qwen's specific chat template directly
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    enc = qwen_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=3000).to(qwen_model.device)
    out = qwen_model.generate(**enc, max_new_tokens=600, do_sample=False, num_beams=1,
                         pad_token_id=qwen_tokenizer.pad_token_id,
                         eos_token_id=qwen_tokenizer.eos_token_id)
                         
    generated = qwen_tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    obj = _extract_json_object(generated)
    
    if not isinstance(obj, dict):
        raise ValueError(f'DP-demo LLM scorer returned invalid JSON: {generated!r}')
    raw = []
    for c in all_cols:
        try:
            val = float(obj.get(c, 0.5))
        except Exception:
            val = 0.5
        raw.append(float(np.clip(val, 0.0, 1.0)))
    raw = np.asarray(raw, dtype=float)
    mean = float(raw.mean())

    normalized = (
        np.ones_like(raw)
        if mean <= 0.0
        else raw / mean
    )

    raw_normalized_weights = {
        c: float(normalized[i])
        for i, c in enumerate(all_cols)
    }

    weights = raw_normalized_weights

    print(
        '[DP-SAMPLE LLM WEIGHTS (QWEN)]',
        cfg.get('name', ''),
        '| raw normalized:',
        raw_normalized_weights,
        '| regularized:',
        weights,
    )

    return weights


def build_weighted_onehot_spec(num_cols, cat_cols, cat_domains, feature_weights):
    """Build deterministic weighted one-hot metadata from public schema domains."""
    spec = {
        'num_cols': list(num_cols),
        'cat_cols': list(cat_cols),
        'cat_domains': {},
        'feature_weights': {
            c: float(feature_weights.get(c, 1.0))
            for c in list(num_cols) + list(cat_cols)
        },
        'feature_slices': {},
        'coordinate_names': [],
        'coordinate_scales': [],
    }
    start = 0
    for col in num_cols:
        weight = max(float(feature_weights.get(col, 1.0)), 0.0)
        scale = float(np.sqrt(weight))
        spec['feature_slices'][col] = slice(start, start + 1)
        spec['coordinate_names'].append(col)
        spec['coordinate_scales'].append(scale)
        start += 1
    for col in cat_cols:
        if col not in cat_domains:
            raise ValueError(f'Missing public categorical domain for: {col}')
        domain = list(map(str, cat_domains[col]))
        if len(domain) == 0:
            raise ValueError(f'Empty public categorical domain for: {col}')
        spec['cat_domains'][col] = domain
        width = len(domain)
        weight = max(float(feature_weights.get(col, 1.0)), 0.0)
        scale = float(np.sqrt(weight / float(width)))
        spec['feature_slices'][col] = slice(start, start + width)
        for category in domain:
            spec['coordinate_names'].append(f'{col}={category}')
            spec['coordinate_scales'].append(scale)
        start += width
    spec['dimension'] = int(start)
    spec['coordinate_scales'] = np.asarray(spec['coordinate_scales'], dtype=float)
    return spec


def encode_weighted_onehot_df(df, spec):
    """Encode a dataframe in the weighted one-hot geometry."""
    X = np.zeros((len(df), int(spec['dimension'])), dtype=float)
    for col in spec['num_cols']:
        sl = spec['feature_slices'][col]
        scale = float(spec['coordinate_scales'][sl.start])
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        X[:, sl.start] = vals * scale
    for col in spec['cat_cols']:
        sl = spec['feature_slices'][col]
        domain = spec['cat_domains'][col]
        mapping = {category: idx for idx, category in enumerate(domain)}
        scale = float(spec['coordinate_scales'][sl.start])
        for row_idx, value in enumerate(df[col].astype(str).tolist()):
            category_idx = mapping.get(value)
            if category_idx is not None:
                X[row_idx, sl.start + category_idx] = scale
    return X


def encode_weighted_onehot_query(row_dict, spec):
    """Encode one query with exactly the same coordinate ordering as training."""
    q = np.zeros(int(spec['dimension']), dtype=float)
    for col in spec['num_cols']:
        sl = spec['feature_slices'][col]
        scale = float(spec['coordinate_scales'][sl.start])
        q[sl.start] = float(np.clip(float(row_dict[col]), 0.0, 1.0)) * scale
    for col in spec['cat_cols']:
        sl = spec['feature_slices'][col]
        domain = spec['cat_domains'][col]
        mapping = {category: idx for idx, category in enumerate(domain)}
        category_idx = mapping.get(str(row_dict[col]))
        if category_idx is not None:
            scale = float(spec['coordinate_scales'][sl.start])
            q[sl.start + category_idx] = scale
    return q


def decode_weighted_onehot_center(center_w, spec):
    """Decode a weighted one-hot DP center for the existing fallback generator."""
    center_w = np.asarray(center_w, dtype=float)
    if len(center_w) != int(spec['dimension']):
        raise ValueError(
            f'Center dimension mismatch: center={len(center_w)}, spec={spec["dimension"]}'
        )
    decoded = []
    for col in spec['num_cols']:
        sl = spec['feature_slices'][col]
        scale = float(spec['coordinate_scales'][sl.start])
        value = 0.5 if scale <= 1e-12 else center_w[sl.start] / scale
        decoded.append(float(np.clip(value, 0.0, 1.0)))
    for col in spec['cat_cols']:
        sl = spec['feature_slices'][col]
        block = center_w[sl]
        decoded.append(int(np.argmax(block)))
    return np.asarray(decoded, dtype=float)


def fit_diffprivlib_dp_kmeans_onehot_weighted(
    df, num_cols, cat_cols, cat_domains, feature_weights,
    k, eps_cluster, random_state,
):
    """Fit DP-KMeans in weighted one-hot space using public-domain bounds."""
    if not HAS_DIFFPRIVLIB_KMEANS:
        raise ImportError(
            f'diffprivlib.models.KMeans is not available. '
            f'Original import error: {DIFFPRIVLIB_IMPORT_ERROR}'
        )
    spec = build_weighted_onehot_spec(
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_domains=cat_domains,
        feature_weights=feature_weights,
    )
    Xw = encode_weighted_onehot_df(df, spec)
    n, d = Xw.shape
    if n == 0:
        raise ValueError('Cannot fit one-hot weighted DP-KMeans on empty dataframe.')
    k_eff = int(min(max(1, k), n))
    lower = np.zeros(d, dtype=float)
    upper = np.maximum(spec['coordinate_scales'], 1e-08)
    model = DPKMeans(
        n_clusters=k_eff,
        epsilon=float(eps_cluster),
        bounds=(lower, upper),
        random_state=int(random_state),
    )
    model.fit(Xw)
    centers_w = np.asarray(model.cluster_centers_, dtype=float)
    labels = np.asarray(model.labels_, dtype=int)
    print(
        '[ONEHOT DP-KMEANS]',
        'shape:', Xw.shape,
        '| K:', k_eff,
        '| counts:', pd.Series(labels).value_counts().sort_index().to_dict(),
    )
    return centers_w, labels, spec


def make_c2adp_centroid_fallback(center, num_cols, cat_cols, cat_domains, n_rows):
    row = {}
    idx = 0
    for col in num_cols:
        row[col] = float(np.clip(center[idx], 0.0, 1.0)) if idx < len(center) else 0.5
        idx += 1
    for col in cat_cols:
        domain = list(map(str, cat_domains.get(col, [])))
        if idx < len(center) and len(domain) > 0:
            raw = float(np.clip(center[idx], 0.0, 1.0))
            cat_idx = int(round(raw * max(len(domain) - 1, 0)))
            cat_idx = int(np.clip(cat_idx, 0, len(domain) - 1))
            row[col] = domain[cat_idx]
        else:
            row[col] = domain[0] if len(domain) > 0 else 'NA'
        idx += 1
    rows = []
    for i in range(int(n_rows)):
        r = dict(row)
        r['target'] = int(i % 2)
        rows.append(r)
    return pd.DataFrame(rows, columns=list(num_cols) + list(cat_cols) + ['target'])


@dataclass
class DatasetContext:
    name: str
    cfg: dict
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    eval_df: pd.DataFrame
    y_true: List[int]
    schema: dict
    cat_domains: Dict[str, Sequence[str]]
    num_bounds: Dict[str, Tuple[float, float]]
    num_cols: List[str]
    cat_cols: List[str]

    @property
    def feature_cols(self) -> List[str]:
        return list(self.num_cols) + list(self.cat_cols)


@dataclass
class BackendAdapter:
    name: str
    global_method: str
    c3_method: str
    global_seed_offset: int
    c3_seed_offset: int
    build_global_pool: Callable[..., pd.DataFrame]
    build_cluster_pools: Callable[..., tuple]
    common_result_metadata: Optional[Mapping[str, Any]] = None


def prepare_dataset(dataset_name: str, eval_seed: int = 777) -> DatasetContext:
    check_data_exists(dataset_name)
    cfg0 = DATASET_META[dataset_name]
    print(f"\n=== DATASET: {dataset_name} ===")

    train_raw = pd.read_csv(cfg0["path"])
    test_raw = pd.read_csv(cfg0["test"])
    train_df, _, _ = prep_dataset(train_raw, cfg0, dataset_name)
    test_df, _, _ = prep_dataset(test_raw, cfg0, dataset_name)

    schema = load_dataset_schema_json(dataset_name)
    cat_domains, num_bounds = schema_to_domains_and_bounds(schema)

    cfg = dict(cfg0)
    cfg.update(
        name=dataset_name,
        schema=schema,
        cat_domains=cat_domains,
        num_bounds=num_bounds,
    )

    feature_cols = [c for c in train_df.columns if c != "target"]
    num_cols = [c for c in feature_cols if c in num_bounds]
    cat_cols = [c for c in feature_cols if c in cat_domains and c not in num_cols]
    omitted = sorted(set(feature_cols) - set(num_cols) - set(cat_cols))
    if omitted:
        raise ValueError(
            f"Public schema does not define these features for {dataset_name}: {omitted}"
        )

    for col in cat_cols:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    if num_cols:
        train_df = clip_and_scale_numeric(train_df, num_cols, num_bounds)
        test_df = clip_and_scale_numeric(test_df, num_cols, num_bounds)

    eval_df = make_balanced_eval(test_df, per_class=50, seed=int(eval_seed))
    if eval_df is None or len(eval_df) == 0:
        raise ValueError(f"No balanced evaluation set could be formed for {dataset_name}.")

    return DatasetContext(
        name=dataset_name,
        cfg=cfg,
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        eval_df=eval_df.reset_index(drop=True),
        y_true=eval_df["target"].astype(int).tolist(),
        schema=schema,
        cat_domains={str(k): list(v) for k, v in cat_domains.items()},
        num_bounds={str(k): tuple(v) for k, v in num_bounds.items()},
        num_cols=list(num_cols),
        cat_cols=list(cat_cols),
    )


def split_c3_partitions(
    train_df: pd.DataFrame,
    sample_frac: float = PAPER_WEIGHT_SAMPLE_FRAC,
    seed: int = PAPER_WEIGHT_SPLIT_SEED,
):
    weight_df, main_df = split_disjoint_weight_sample(
        train_df=train_df,
        sample_frac=float(sample_frac),
        seed=int(seed),
    )
    return weight_df.reset_index(drop=True), main_df.reset_index(drop=True)


def _validate_cached_weight_payload(
    payload: dict,
    cache_path: Path,
    epsilon: float,
    split_seed: int,
    expected_features: set,
):
    if "normalized_weights" not in payload:
        raise ValueError(f"Missing normalized_weights in {cache_path}")
    weights = {str(k): float(v) for k, v in payload["normalized_weights"].items()}
    if float(payload.get("epsilon")) != float(epsilon):
        raise ValueError(f"Epsilon mismatch in {cache_path}")
    if int(payload.get("weight_split_seed")) != int(split_seed):
        raise ValueError(f"Split-seed mismatch in {cache_path}")
    if set(weights) != expected_features:
        raise ValueError(f"Feature mismatch in {cache_path}")
    return weights


def load_or_build_task_guided_weights(
    ctx: DatasetContext,
    epsilon: float,
    weight_df: pd.DataFrame,
    cache_dir: str = "qwen_dp_weights_cache",
    split_fraction: float = PAPER_WEIGHT_SAMPLE_FRAC,
    split_seed: int = PAPER_WEIGHT_SPLIT_SEED,
    summary_base_seed: int = PAPER_WEIGHT_SUMMARY_BASE_SEED,
):
    cache_root = Path(cache_dir)
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_path = cache_root / f"{ctx.name.lower()}_eps_{float(epsilon):g}.json"
    summary_seed = int(summary_base_seed) + int(round(float(epsilon) * 1000))
    expected_features = set(ctx.feature_cols)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        weights = _validate_cached_weight_payload(
            payload, cache_path, epsilon, split_seed, expected_features
        )
        print(f"[C3 WEIGHTS CACHE] loaded {cache_path}")
        return weights, str(cache_path)

    rng = np.random.RandomState(summary_seed)
    dp_demos = build_dp_weighting_demos(
        weight_df=weight_df,
        num_cols=ctx.num_cols,
        cat_cols=ctx.cat_cols,
        epsilon=float(epsilon),
        rng=rng,
        cat_domains=ctx.cat_domains,
        dataset_name=ctx.name,
        num_bounds=ctx.num_bounds,
    )
    weights = build_llm_weights_from_dp_demos(
        cfg=ctx.cfg,
        dp_demos=dp_demos,
        num_cols=ctx.num_cols,
        cat_cols=ctx.cat_cols,
        schema=ctx.schema,
    )
    weights = {str(k): float(v) for k, v in weights.items()}
    if set(weights) != expected_features:
        raise ValueError(
            f"Expected features {sorted(expected_features)}, received {sorted(weights)}"
        )

    payload = {
        "dataset": ctx.name,
        "epsilon": float(epsilon),
        "weight_sample_fraction": float(split_fraction),
        "weight_split_seed": int(split_seed),
        "summary_seed": int(summary_seed),
        "scoring_model": QWEN_MODEL_NAME,
        "prompt_version": "qwen_dp_summary_weights_v1",
        "dp_summaries": dp_demos.to_dict(orient="records"),
        "normalized_weights": weights,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[C3 WEIGHTS CACHE] saved {cache_path}")
    return weights, str(cache_path)


def route_to_cluster_pool(row_dict, centers_w, stores, onehot_spec):
    qvec = encode_weighted_onehot_query(row_dict, onehot_spec)
    valid = [
        int(cid)
        for cid, pool in stores.items()
        if pool is not None and len(pool) > 0
    ]
    if not valid:
        raise RuntimeError("No non-empty cluster pool is available for routing.")
    best_cluster = min(
        valid,
        key=lambda cid: float(np.sum((qvec - np.asarray(centers_w[cid])) ** 2)),
    )
    return int(best_cluster), stores[best_cluster]


def evaluate_pool_selector(
    ctx: DatasetContext,
    trial_seed: int,
    pool_selector: Callable[[dict], Tuple[Optional[int], pd.DataFrame]],
    shots: int = PAPER_SHOTS,
):
    predictions: List[int] = []
    retrieval_rows: List[dict] = []

    for row_idx, (_, row) in enumerate(ctx.eval_df.iterrows()):
        row_dict = {col: row[col] for col in ctx.feature_cols}
        selected_cluster, pool = pool_selector(row_dict)
        if pool is None or len(pool) == 0:
            raise RuntimeError(f"Query {row_idx}: selected demonstration pool is empty.")

        row_seed = int(trial_seed) * 10000 + int(row_idx)
        demo_df = sample_balanced_shots(
            pool,
            k=int(shots),
            seed=row_seed,
            label_col="target",
        )
        pred = execute_classification_strategy(
            ctx.cfg,
            row_dict,
            demo_df,
            ctx.feature_cols,
        )
        predictions.append(int(pred))
        retrieval_rows.append(
            {
                "query_index": int(row_idx),
                "true_label": int(row["target"]),
                "prediction": int(pred),
                "selected_cluster": selected_cluster,
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(ctx.y_true, predictions)),
        "f1": float(f1_score(ctx.y_true, predictions, zero_division=0)),
        "precision": float(precision_score(ctx.y_true, predictions, zero_division=0)),
        "recall": float(recall_score(ctx.y_true, predictions, zero_division=0)),
    }
    return metrics, retrieval_rows


def fallback_count(logs) -> int:
    if not isinstance(logs, Mapping):
        return 0
    return int(
        sum(
            1
            for value in logs.values()
            if isinstance(value, Mapping) and bool(value.get("fallback", False))
        )
    )


def append_results(results_csv: str, result: dict, retrieval_rows: Sequence[dict]):
    append_row_to_csv(result, results_csv)
    retrieval_path = str(Path(results_csv).with_suffix("")) + "_retrieval.csv"
    for row in retrieval_rows:
        append_row_to_csv(dict(row), retrieval_path)


def run_backend_experiments(
    adapter: BackendAdapter,
    datasets: Sequence[str],
    methods: Sequence[str],
    trials: int,
    k_list: Sequence[int],
    eps_list: Sequence[float],
    results_csv: str,
    cluster_share: float = PAPER_CLUSTER_SHARE,
    delta: float = PAPER_DELTA,
    weight_sample_frac: float = PAPER_WEIGHT_SAMPLE_FRAC,
    weight_split_seed: int = PAPER_WEIGHT_SPLIT_SEED,
    weight_summary_base_seed: int = PAPER_WEIGHT_SUMMARY_BASE_SEED,
    weight_cache_dir: str = "qwen_dp_weights_cache",
    shots: int = PAPER_SHOTS,
    eval_seed: int = 777,
):
    allowed = {adapter.global_method, adapter.c3_method}
    methods = list(methods)
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    if not 0.0 < float(cluster_share) < 1.0:
        raise ValueError("cluster_share must lie in (0, 1).")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    use_c3 = adapter.c3_method in methods

    for dataset_name in datasets:
        ctx = prepare_dataset(dataset_name, eval_seed=eval_seed)
        n_synth_global = int(len(ctx.train_df))

        weight_df = main_df = None
        if use_c3:
            weight_df, main_df = split_c3_partitions(
                ctx.train_df,
                sample_frac=float(weight_sample_frac),
                seed=int(weight_split_seed),
            )
            print(
                "[C3 DISJOINT SPLIT]",
                "dataset=", ctx.name,
                "| split_seed=", int(weight_split_seed),
                "| weight_rows=", len(weight_df),
                "| main_rows=", len(main_df),
            )

        for k in k_list:
            k = int(k)
            rows_per_cluster = max(1, n_synth_global // max(k, 1))

            for epsilon in eps_list:
                epsilon = float(epsilon)
                eps_cluster = float(cluster_share) * epsilon
                eps_synth = epsilon - eps_cluster

                c3_weights = None
                c3_cache_path = None
                if use_c3:
                    c3_weights, c3_cache_path = load_or_build_task_guided_weights(
                        ctx=ctx,
                        epsilon=epsilon,
                        weight_df=weight_df,
                        cache_dir=weight_cache_dir,
                        split_fraction=float(weight_sample_frac),
                        split_seed=int(weight_split_seed),
                        summary_base_seed=int(weight_summary_base_seed),
                    )
                    print(
                        "[C3 DP-GUIDED WEIGHTS]",
                        ctx.name,
                        "| epsilon=", epsilon,
                        "| cache=", c3_cache_path,
                        "| weights=", c3_weights,
                    )

                for trial in range(int(trials)):
                    trial_seed = 1000 + int(trial)
                    print(f"\n[{ctx.name}] K={k} eps={epsilon} seed={trial_seed}")

                    global_pool = None
                    centers = stores = logs = onehot_spec = None

                    if adapter.global_method in methods:
                        global_seed = trial_seed + int(adapter.global_seed_offset)
                        set_seed(global_seed)
                        global_pool = adapter.build_global_pool(
                            ctx=ctx,
                            epsilon=epsilon,
                            delta=float(delta),
                            n_synth=n_synth_global,
                            seed=global_seed,
                        )

                    if use_c3:
                        method_seed = trial_seed + int(adapter.c3_seed_offset)
                        set_seed(method_seed)
                        centers, stores, logs, onehot_spec = adapter.build_cluster_pools(
                            ctx=ctx,
                            main_df=main_df,
                            feature_weights=c3_weights,
                            k=k,
                            eps_cluster=eps_cluster,
                            eps_synth=eps_synth,
                            delta=float(delta),
                            rows_per_cluster=rows_per_cluster,
                            seed=method_seed,
                        )

                    for method in methods:
                        if method == adapter.global_method:
                            def selector(_row_dict, _pool=global_pool):
                                return None, _pool
                        elif method == adapter.c3_method:
                            def selector(
                                row_dict,
                                _centers=centers,
                                _stores=stores,
                                _spec=onehot_spec,
                            ):
                                return route_to_cluster_pool(
                                    row_dict, _centers, _stores, _spec
                                )
                        else:
                            raise AssertionError(method)

                        metrics, retrieval_rows = evaluate_pool_selector(
                            ctx=ctx,
                            trial_seed=trial_seed,
                            pool_selector=selector,
                            shots=shots,
                        )

                        is_c3 = method == adapter.c3_method
                        result = {
                            "dataset": ctx.name,
                            "method": method,
                            "epsilon": epsilon,
                            "delta": float(delta),
                            "K": k,
                            "seed": int(trial_seed),
                            **metrics,
                            "base_synthesizer": adapter.name,
                            "uses_dp_summaries": bool(is_c3),
                            "weight_source": "dp_summary_qwen_disjoint_c3" if is_c3 else "none",
                            "eps_cluster": float(eps_cluster) if is_c3 else 0.0,
                            "eps_synth": float(eps_synth) if is_c3 else epsilon,
                            "target_centered": bool(is_c3),
                            "fallback_clusters": fallback_count(logs) if is_c3 else 0,
                            "weight_sample_fraction": float(weight_sample_frac) if is_c3 else 0.0,
                            "weight_split_seed": int(weight_split_seed) if is_c3 else None,
                            "weight_cache_path": c3_cache_path if is_c3 else None,
                            "main_partition_rows": int(len(main_df)) if is_c3 else int(len(ctx.train_df)),
                        }
                        if adapter.common_result_metadata:
                            result.update(dict(adapter.common_result_metadata))

                        tagged_retrieval = []
                        for item in retrieval_rows:
                            item = dict(item)
                            item.update(
                                dataset=ctx.name,
                                method=method,
                                epsilon=epsilon,
                                K=k,
                                seed=int(trial_seed),
                            )
                            tagged_retrieval.append(item)

                        append_results(results_csv, result, tagged_retrieval)
                        print(
                            f"[RESULT] {ctx.name} {method} eps={epsilon} "
                            f"K={k} seed={trial_seed} F1={metrics['f1']:.4f}"
                        )

                    cleanup()
