import os, gc, random, argparse, re, json, hashlib
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from snsynth import Synthesizer
from snsynth.transform import TableTransformer, MinMaxTransformer, ChainTransformer, LabelTransformer, OneHotEncoder
try:
    from diffprivlib.models import KMeans as DPKMeans
    HAS_DIFFPRIVLIB_KMEANS = True
except Exception as e:
    DPKMeans = None
    HAS_DIFFPRIVLIB_KMEANS = False
    DIFFPRIVLIB_IMPORT_ERROR = e
os.environ["JAX_PLATFORM_NAME"] = "gpu"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['VECLIB_MAXIMUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ['NUMBA_NUM_THREADS'] = '16'
THREAT_MODEL = 'trusted_curator'
DEBUG_GENERATE = True
DP_PRIVATE_DEBUG = False
SERIALIZATION = 'json'
UNWEIGHTED_ABLATION = False

# These shares are inherited from the bundled PrivSyn Anonymisation.py:
# 10% of rho for one-way marginals and 80% for publication of the selected
# multi-way workload. The remaining portion is used by PrivSyn's private
# dependency-selection stage. They are fixed properties of the base method,
# not tunable PriTabICL hyperparameters.
PRIVSYN_ONE_WAY_RHO_SHARE = 0.10
PRIVSYN_MARGINAL_PUBLICATION_RHO_SHARE = 0.80
PUBLIC_RELEVANCE_CACHE_DIR = 'public_semantic_relevance'
PUBLIC_RELEVANCE_PROMPT_VERSION = 'llm_score_v3'

def load_dataset_schema_json(dataset_name, schema_dir=None):
    fname_map = {'Adult': 'adult.json', 'Magic': 'magic.json', 'Phishing': 'phishing.json', 'Abalone': 'abalone.json', 'Shoppers': 'shoppers.json', 'Diabetes': 'diabetes.json', 'DefaultCredit': 'dcredit.json', 'Banking': 'banking.json', 'Airline': 'airline.json', 'Weather': 'weather.json', 'ACSPublicCoverage': 'publiccoverage_enhanced.json', 'Mushroom': 'mushroom.json','Heart': 'heart.json'}
    target = fname_map.get(dataset_name, None)
    if target is None:
        return None
    if schema_dir is None:
        ds_lower = dataset_name.lower()
        candidates = [os.path.join(os.getcwd(), 'schemas'), os.getcwd(), os.path.join(os.getcwd(), 'datasets', ds_lower), os.path.join(os.getcwd(), 'datasets', ds_lower.replace(' ', ''))]
        if dataset_name in DATASET_META:
            meta_dir = os.path.dirname(DATASET_META[dataset_name]['path'])
            candidates.append(meta_dir)
    else:
        candidates = [schema_dir]
    for base in candidates:
        for fname in (target, target.lower(), target.upper()):
            path = os.path.join(base, fname)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
    return None

def append_row_to_csv(row: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df_row = pd.DataFrame([row])
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    df_row.to_csv(csv_path, mode='a', header=not exists, index=False)

def combine_cluster_store(synth_store, cluster_col='cluster_id'):
    if synth_store is None:
        return pd.DataFrame()
    parts = []
    for cid, df_c in synth_store.items():
        if df_c is None or len(df_c) == 0:
            continue
        tmp = df_c.copy()
        tmp[cluster_col] = int(cid) if not isinstance(cid, tuple) else str(cid)
        parts.append(tmp)
    if len(parts) == 0:
        return pd.DataFrame()
    return pd.concat(parts, axis=0).reset_index(drop=True)

def save_synth_pool_csv(out_root, dataset_name, method, epsilon, K, seed, synth_df, num_cols, cat_cols, extra_meta=None):
    run_dir = os.path.join(out_root, f'dataset={dataset_name}', f'eps={epsilon}', f'K={K}', f'seed={seed}', f'method={method}')
    os.makedirs(run_dir, exist_ok=True)
    if synth_df is None:
        synth_df = pd.DataFrame()
    synth_df = synth_df.copy().reset_index(drop=True)
    synth_path = os.path.join(run_dir, f'{method}_synth_all.csv')
    synth_df.to_csv(synth_path, index=False)
    if 'cluster_id' in synth_df.columns:
        for cid, df_c in synth_df.groupby('cluster_id'):
            df_c.to_csv(os.path.join(run_dir, f'{method}_synth_cluster_{cid}.csv'), index=False)
    summary = {'dataset': dataset_name, 'method': method, 'epsilon': epsilon, 'K': K, 'seed': seed, 'n_synth': len(synth_df)}
    if 'target' in synth_df.columns and len(synth_df) > 0:
        counts = synth_df['target'].value_counts(dropna=False).to_dict()
        summary['label_counts'] = json.dumps({str(k): int(v) for k, v in counts.items()})
        summary['positive_rate'] = float(synth_df['target'].astype(int).mean())
    if extra_meta is not None:
        summary.update(extra_meta)
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, f'{method}_summary.csv'), index=False)
    feature_rows = []
    for c in num_cols:
        if c in synth_df.columns and len(synth_df) > 0:
            vals = pd.to_numeric(synth_df[c], errors='coerce')
            feature_rows.append({'feature': c, 'type': 'numeric', 'mean': vals.mean(), 'std': vals.std(), 'min': vals.min(), 'max': vals.max()})
    for c in cat_cols:
        if c in synth_df.columns and len(synth_df) > 0:
            vc = synth_df[c].astype(str).value_counts(normalize=True, dropna=False).head(10)
            feature_rows.append({'feature': c, 'type': 'categorical', 'top_values': json.dumps({str(k): float(v) for k, v in vc.to_dict().items()})})
    pd.DataFrame(feature_rows).to_csv(os.path.join(run_dir, f'{method}_feature_summary.csv'), index=False)
    if 'cluster_id' in synth_df.columns and len(synth_df) > 0:
        cluster_rows = []
        for cid, df_c in synth_df.groupby('cluster_id'):
            row = {'cluster_id': cid, 'n_synth': len(df_c)}
            if 'target' in df_c.columns:
                counts = df_c['target'].value_counts(dropna=False).to_dict()
                row['label_counts'] = json.dumps({str(k): int(v) for k, v in counts.items()})
                row['positive_rate'] = float(df_c['target'].astype(int).mean())
            cluster_rows.append(row)
        pd.DataFrame(cluster_rows).to_csv(os.path.join(run_dir, f'{method}_cluster_summary.csv'), index=False)
    print(f'[SAVED SYNTH ANALYSIS] {method} -> {run_dir}')

def save_retrieval_log_csv(out_root, dataset_name, method, epsilon, K, seed, retrieval_rows):
    if retrieval_rows is None or len(retrieval_rows) == 0:
        return
    run_dir = os.path.join(out_root, f'dataset={dataset_name}', f'eps={epsilon}', f'K={K}', f'seed={seed}', f'method={method}')
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(retrieval_rows).to_csv(os.path.join(run_dir, f'{method}_retrieval_log.csv'), index=False)
    print(f'[SAVED RETRIEVAL LOG] {method} -> {run_dir}')

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

def unscale_numeric_value(x01, L, U):
    return float(x01) * (U - L) + L

def get_cat_domains(cfg, train_df, cat_cols, allow_private_fallback=False):
    domains = {}
    cfg_domains = cfg.get('cat_domains', None) if isinstance(cfg, dict) else None
    if cfg_domains:
        for c in cat_cols:
            if c in cfg_domains:
                domains[c] = list(map(str, cfg_domains[c]))
    missing = [c for c in cat_cols if c not in domains]
    if missing and (not allow_private_fallback):
        raise ValueError('Missing categorical domains for columns: %s. Provide schema-derived cat_domains (recommended).' % missing)
    if missing and allow_private_fallback:
        print('[WARN] Falling back to train_df.unique() domains for:', missing)
        for c in missing:
            domains[c] = list(map(str, sorted(train_df[c].astype(str).unique().tolist())))
    return domains
DEBUG_PROMPT = False
DEBUG_PROMPT_ONCE = False
_prompt_debug_printed = False
DATASETS_TO_RUN = ['Adult', 'Magic', 'Phishing', 'Abalone', 'Shoppers']
N_TRIALS = 20
SHOT_COUNTS = [8]
EVAL_PER_CLASS = 50
K_LIST = [8]
CLUSTER_SEED = 42
TOP_R_CLUSTERS = 1
PUBLIC_LABELS = [0, 1]
N_SYN_GLOBAL = 7891
N_SYN_PER_LABEL = 10000
MIN_CLUSTER_SIZE = 100
METHODS = ['C3_PRIVSYN_ICL_DP_SAMPLE_LLM']
RESULTS_CSV = 'advisor_protocol_results.csv'
PLOTS_DIR = 'plots'
SAVE_SYNTH_ANALYSIS = False
SYNTH_ANALYSIS_DIR = 'synthetic_analysis_outputs3070'
MODEL_NAME = 'meta-llama/Llama-2-13b-chat-hf'
QWEN_MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'
METHOD_SEED_OFFSETS = {'A2': 11, 'B1': 101, 'GDP': 151, 'C1': 202, 'C3_GLOBAL_PRIVSYN': 858, 'C3_GLOBAL_PRIVSYN_TA_RETRIEVE': 868, 'C3_PRIVSYN_ICL_CONSTRUCT': 969, 'C3_PRIVSYN_ICL_DP_SAMPLE_LLM': 1081,'C3_PRIVSYN_ICL_UNWEIGHTED': 1081,'C3_GLOBAL_PRIVSYN_TARGET_ONLY': 858, 'C3_PRIVSYN_ICL_DP_SAMPLE_UNWEIGHTED': 1081,}
WEIGHT_SAMPLE_FRAC = 0.10
WEIGHT_DP_SHOTS = 2

# The 10%/90% data partition stays identical for every epsilon and trial.
WEIGHT_SPLIT_SEED = 314159

# The DP summary noise changes with epsilon, but not with trial.
WEIGHT_SUMMARY_BASE_SEED = 271828
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    safe = {v: counts.get(v, 0) + rng.laplace(0, 1.0 / epsilon) for v in possible_values}
    return max(safe, key=safe.get)
CAREY_SUPPORTED_K = [1, 2, 4, 8]
CAREY_NUMERIC_THRESHOLDS_ORIGINAL = {'Diabetes': {'Pregnancies': 4, 'Age': 33}}

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
        # For datasets not covered by the DP-TabICL paper, this is an explicit
        # public-schema adaptation rather than an exact reproduction.
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

def seed_for_method(trial_seed, method_name):
    offset = METHOD_SEED_OFFSETS.get(method_name, 999)
    method_seed = int(trial_seed + offset)
    set_seed(method_seed)
    return method_seed

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

def strat_GDP_tabicl(cfg, row_dict, gdp_demos, num_cols, cat_cols):
    if gdp_demos is None or len(gdp_demos) == 0:
        return 0
    shots = gdp_demos.copy().reset_index(drop=True)
    print('[DEBUG GDP] shot label counts:', shots['target'].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, num_cols + cat_cols)

def mixed_dist(row_dict, cand, num_cols, cat_cols):
    d_num = 0.0
    for c in num_cols:
        d_num += abs(float(row_dict[c]) - float(cand[c]))
    d_cat = 0.0
    for c in cat_cols:
        d_cat += 1.0 if str(row_dict[c]) != str(cand[c]) else 0.0
    return d_num + d_cat

def compute_internal_distortion(df, clusters, dp_centroids, num_cols, cat_cols):
    distortions = []
    for c in np.unique(clusters):
        rows = df[clusters == c]
        if rows.empty or c not in dp_centroids:
            continue
        true_cent = {col: rows[col].mean() for col in num_cols}
        for col in cat_cols:
            true_cent[col] = rows[col].mode()[0]
        distortions.append(mixed_dist(true_cent, dp_centroids[c], num_cols, cat_cols))
    return float(np.mean(distortions)) if distortions else 0.0

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

def _apply_numeric_binning(df, num_cols, bin_edges):
    df = df.copy()
    for c in num_cols:
        vals = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(0.0, 1.0).to_numpy()
        e = bin_edges[c]
        idx = np.digitize(vals, e[1:-1], right=False)
        df[c] = idx.astype(str)
    return df

def _bin_ids_to_midpoints(series, edges):
    mids = (edges[:-1] + edges[1:]) / 2.0
    idx = pd.to_numeric(series, errors='coerce').fillna(0).astype(int).clip(0, len(mids) - 1)
    return mids[idx]

def _decode_binned_numeric(df, num_cols, bin_edges):
    df = df.copy()
    for c in num_cols:
        df[c] = _bin_ids_to_midpoints(df[c], bin_edges[c])
    return df
DATASET_META = {'Adult': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/adult/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/adult/test.csv', 'label': 'label', 'pos': '>50K', 'task': 'income prediction', 'decode_prompt_values': True}, 
        'Magic': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/magic/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/magic/test.csv', 'label': 'label', 'pos': 'g', 'task': 'particle classification'}, 
        'Phishing': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/phishing/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/phishing/test.csv', 'label': 'label', 'pos': '1', 'task': 'phishing website detection', 'decode_prompt_values': False}, 
        'Shoppers': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/shoppers/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/shoppers/test.csv', 'label': 'label', 'pos': 'TRUE', 'task': 'online purchase prediction', 'decode_prompt_values': True}, 
        'Abalone': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/Abalone/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/Abalone/test.csv', 'label': 'label', 'pos': '10', 'task': 'abalone age classification'}, 
        'Diabetes': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/diabetes/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/diabetes/test.csv', 'label': 'Diabetes_binary', 'pos': '1', 'task': 'diabetes risk prediction', 'decode_prompt_values': False},
        'DefaultCredit': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/dcredit/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/dcredit/test.csv', 'label': 'label', 'pos': '1', 'task': 'credit card default prediction'}, 
        'Banking': {'path': '/content/drive/MyDrive/Colab Notebooks/datasets/banking/train.csv', 'test': '/content/drive/MyDrive/Colab Notebooks/datasets/banking/test.csv', 'label': 'y', 'pos': '1', 'task': 'bank term deposit prediction', 'decode_prompt_values': True},  
        'Mushroom': {
        'path': '/content/drive/MyDrive/Colab Notebooks/datasets/mushroom/train.csv',
        'test': '/content/drive/MyDrive/Colab Notebooks/datasets/mushroom/test.csv',
        'label': 'Mushroom_quality',
        'pos': 'p',
        'task': 'poisonous mushroom classification',
        'decode_prompt_values': True
    },
    'Airline': {
        'path': '/content/drive/MyDrive/Colab Notebooks/datasets/airline/train.csv', 
        'test': '/content/drive/MyDrive/Colab Notebooks/datasets/airline/test.csv', 
        'label': 'satisfaction', 
        'pos': 'satisfied', 
        'task': 'passenger satisfaction prediction',
        'decode_prompt_values': True
    },
    'Weather': {
        'path': '/content/drive/MyDrive/Colab Notebooks/datasets/weather/train.csv', 
        'test': '/content/drive/MyDrive/Colab Notebooks/datasets/weather/test.csv', 
        'label': 'RainTomorrow', 
        'pos': 'Yes', 
        'task': 'rain prediction'
    },
    'ACSPublicCoverage': {
        'path': '/content/drive/MyDrive/Colab Notebooks/datasets/pcoverage/train.csv',  
        'test': '/content/drive/MyDrive/Colab Notebooks/datasets/pcoverage/test.csv',   
        'label': 'label',                   
        'pos': '1',                         
        'task': 'Predict whether a U.S. resident under age 65 with personal income at or below $30,000 is covered by public health insurance.',
        'decode_prompt_values': True
    },
    'Heart': {
    'path': '/content/drive/MyDrive/Colab Notebooks/datasets/heart/train.csv',
    'test': '/content/drive/MyDrive/Colab Notebooks/datasets/heart/test.csv',
    'label': 'HeartDisease',
    'pos': '1',
    'task': 'Predict whether a patient has heart disease based on demographic and clinical measurements.',
    'decode_prompt_values': True
}
    
}

def check_data_exists(dataset_name):
    cfg = DATASET_META[dataset_name]
    if not os.path.exists(cfg['path']) or not os.path.exists(cfg['test']):
        raise FileNotFoundError(f"\n\n🚨 ERROR: Could not find the dataset files for {dataset_name}!\nExpected paths:\n  Train: {cfg['path']}\n  Test:  {cfg['test']}\n--> Action Required: Please upload your 'datasets' folder to this environment.\n")

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
print(f'Loading: {MODEL_NAME}')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.truncation_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', quantization_config=bnb_config, low_cpu_mem_usage=True, attn_implementation='sdpa')
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

print('Loading Weighting Model: Qwen/Qwen2.5-14B-Instruct')
qwen_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct', use_fast=True)
if qwen_tokenizer.pad_token is None:
    qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

qwen_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-14B-Instruct', 
    device_map='auto', 
    quantization_config=bnb_config, 
    low_cpu_mem_usage=True, 
    attn_implementation='sdpa'
)
qwen_model.config.pad_token_id = qwen_tokenizer.pad_token_id
qwen_model.eval()

@torch.no_grad()
def _make_chat_prompt(system_msg: str, user_msg: str) -> str:
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system_msg + '\n\n' + user_msg + '\nAnswer: '

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

def format_row(row_dict: dict, cols: list) -> str:
    return '\n'.join([f'- {c}: {format_value(row_dict.get(c))}' for c in cols])

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

@torch.inference_mode()
def llm_predict_binary(system_msg: str, user_msg: str) -> int:
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

def retrieve_balanced_nearest_shots(pool_df, row_dict, k, num_cols, cat_cols, seed=None, label_col='target'):
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
        out['_dist'] = out.apply(lambda r: mixed_dist(row_dict, {c: r[c] for c in num_cols + cat_cols}, num_cols, cat_cols), axis=1)
        return out
    pos = _add_dist(pos)
    neg = _add_dist(neg)
    k_pos = k // 2
    k_neg = k - k_pos
    pos_shots = pos.sort_values('_dist').head(k_pos) if len(pos) > 0 else pos
    neg_shots = neg.sort_values('_dist').head(k_neg) if len(neg) > 0 else neg
    shots = pd.concat([pos_shots, neg_shots], axis=0)
    remaining = k - len(shots)
    if remaining > 0:
        used_idx = set(shots.index.tolist())
        leftover = pd.concat([pos, neg], axis=0)
        leftover = leftover[~leftover.index.isin(used_idx)].sort_values('_dist')
        fill = leftover.head(remaining)
        shots = pd.concat([shots, fill], axis=0)
    if seed is not None and len(shots) > 0:
        shots = shots.sample(frac=1.0, random_state=seed)
    return shots.drop(columns=['_dist'], errors='ignore').reset_index(drop=True)

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

def retrieve_nearest_shots(pool_df, row_dict, k, num_cols, cat_cols, seed=None):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None
    df = pool_df.copy()

    def _dist(r):
        cand = {c: r[c] for c in num_cols + cat_cols}
        return mixed_dist(row_dict, cand, num_cols, cat_cols)
    df['_dist'] = df.apply(_dist, axis=1)
    shots = df.sort_values('_dist').head(k).copy()
    if seed is not None:
        shots = shots.sample(frac=1.0, random_state=seed)
    return shots.drop(columns=['_dist'], errors='ignore').reset_index(drop=True)

def build_clean_pool(train_df, k_pool=2000, seed=42):
    return train_df.copy().reset_index(drop=True) if len(train_df) <= k_pool else train_df.sample(k_pool, random_state=seed).reset_index(drop=True)

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

def encode_features_for_dp_kmeans(df, num_cols, cat_cols, cat_domains):
    parts = []
    if len(num_cols) > 0:
        X_num = df[num_cols].to_numpy(dtype=float)
        X_num = np.clip(X_num, 0.0, 1.0)
        parts.append(X_num)
    for c in cat_cols:
        if c not in cat_domains:
            raise ValueError(f'Missing public categorical domain for column: {c}')
        domain = list(map(str, cat_domains[c]))
        mapping = {v: i for i, v in enumerate(domain)}
        vals = df[c].astype(str).map(mapping).fillna(0).to_numpy(dtype=float)
        denom = max(len(domain) - 1, 1)
        vals = (vals / denom).reshape(-1, 1)
        parts.append(vals)
    if len(parts) == 0:
        raise ValueError('No features available for DP-KMeans.')
    return np.concatenate(parts, axis=1)

def encoded_query_vector(row_dict, num_cols, cat_cols, cat_domains):
    vals = []
    for c in num_cols:
        vals.append(float(np.clip(float(row_dict[c]), 0.0, 1.0)))
    for c in cat_cols:
        if c not in cat_domains:
            raise ValueError(f'Missing public categorical domain for column: {c}')
        domain = list(map(str, cat_domains[c]))
        mapping = {v: i for i, v in enumerate(domain)}
        idx = mapping.get(str(row_dict[c]), 0)
        denom = max(len(domain) - 1, 1)
        vals.append(float(idx) / denom)
    return np.asarray(vals, dtype=float)

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


@torch.inference_mode()
def _llm_score_batch_public_features(task_description, features_dict):
    system_msg = (
        "You are an expert data scientist assessing feature relevance for a "
        "tabular binary-classification task. Evaluate ALL candidate features "
        "jointly and rank them relative to one another based only on their "
        "semantic meaning and the stated prediction task. "
        "\n\n"
        "Assign each feature a score from 0.0 to 1.0:\n"
        "- 0.0 to 0.1: irrelevant, identifier, metadata, or likely noise\n"
        "- 0.2 to 0.3: weak or indirect predictor\n"
        "- 0.4 to 0.6: moderately useful predictor\n"
        "- 0.7 to 0.8: strong predictor directly related to the target\n"
        "- 0.9 to 1.0: among the most critical predictors\n"
        "\n"
        "Return ONLY a valid JSON dictionary mapping every exact feature name "
        "to one floating-point score. Include every candidate feature exactly "
        "once. Do not provide explanations or markdown. Start with \"{\" and "
        "end with \"}\"."
    )

    features_text = "\n".join(
        f"- {name}: {description}"
        for name, description in features_dict.items()
    )

    user_msg = (
        f"Prediction task:\n{task_description}\n\n"
        f"Candidate features:\n{features_text}\n\n"
        "Return only a JSON dictionary such as:\n"
        '{"direct predictor": 0.9, "moderate predictor": 0.5, '
        '"weak predictor": 0.2}'
    )
        
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    enc = qwen_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(qwen_model.device)
    
    out = qwen_model.generate(
        **enc,
        max_new_tokens=1200,
        do_sample=False,
        num_beams=1,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=qwen_tokenizer.eos_token_id,
    )
    generated = qwen_tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    obj = _extract_json_object(generated)
    
    if obj is None or not isinstance(obj, dict):
        raise ValueError(f'Public semantic scorer returned invalid JSON: {generated!r}')
    
    return obj
    
def sample_random_shots(
    pool_df: pd.DataFrame,
    k: int,
    seed: int,
):
    if pool_df is None or len(pool_df) == 0 or k <= 0:
        return None

    k = int(k)
    replace = k > len(pool_df)

    return (
        pool_df.sample(
            n=k,
            replace=replace,
            random_state=seed,
        )
        .reset_index(drop=True)
    )
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


@torch.inference_mode()
def _llm_score_one_public_feature(task_description, feature_description):
    system_msg = (
        'You score the semantic relevance of one tabular feature for a prediction task. '
        'Use general domain knowledge only. Do not assume access to dataset rows, labels, '
        'empirical correlations, distributions, frequencies, or evaluation results. '
        'Return JSON only. Do not include any explanations, preambles, or conversational text. Start your response with "{".'
    )
    user_msg = (
        f'Prediction task: {task_description}\n'
        f'Candidate feature: {feature_description}\n'
        'Return exactly {"score": x}, where x is a real number between 0 and 1. '
        'A larger score means stronger expected relevance to the target.'
    )
    
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    enc = qwen_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(qwen_model.device)
    out = qwen_model.generate(
        **enc,
        max_new_tokens=150,
        do_sample=False,
        num_beams=1,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=qwen_tokenizer.eos_token_id,
    )
    generated = qwen_tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
    obj = _extract_json_object(generated)
    
    if obj is None or 'score' not in obj:
        raise ValueError(f'Public semantic scorer returned invalid JSON: {generated!r}')
    score = float(obj['score'])
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(f'Public semantic score must be in [0,1], got {score!r}')
    return score


def build_taskaware_feature_weights(cfg, num_cols, cat_cols, schema=None, cache_dir=PUBLIC_RELEVANCE_CACHE_DIR):
    """Construct data-independent semantic relevance scores.

    This follows the LLM-Score setting: each feature is scored from only its
    public name/type and the public prediction task. The LLM never sees private
    rows or empirical statistics. Scores are cached and reused for every seed,
    epsilon, and method invocation.

    Returned values are mean-normalized relevance coefficients r_j. Clustering
    uses sqrt(r_j) as a coordinate multiplier so squared Euclidean distance has
    coefficient r_j exactly. Marginal selection uses r_j directly.
    """
    all_cols = list(num_cols) + list(cat_cols)
    if not all_cols:
        return {}
    task_description = _public_target_description(cfg, schema)
    public_payload = {
        'version': PUBLIC_RELEVANCE_PROMPT_VERSION,
        'model': QWEN_MODEL_NAME,
        'dataset': str(cfg.get('name', 'dataset')),
        'task': task_description,
        'features': [_public_feature_description(c, schema, num_cols, cat_cols) for c in all_cols],
    }
    payload_hash = hashlib.sha256(json.dumps(public_payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{str(cfg.get('name','dataset')).lower()}_{payload_hash}.json")
    raw_scores = None
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        raw_scores = cached.get('raw_scores')
        if not isinstance(raw_scores, dict) or set(raw_scores) != set(all_cols):
            raise ValueError(f'Invalid public relevance cache: {cache_path}')
    else:
        features_dict = {}
        for feature in all_cols:
            features_dict[feature] = _public_feature_description(feature, schema, num_cols, cat_cols)
        
        raw_scores_str = _llm_score_batch_public_features(task_description, features_dict)
        
        raw_scores = {}
        for feature in all_cols:
            if feature not in raw_scores_str:
                print(f"[WARN] Feature '{feature}' missing from LLM batch output, defaulting to 0.5")
                raw_scores[feature] = 0.5
            else:
                try:
                    score = float(raw_scores_str[feature])
                    raw_scores[feature] = max(0.0, min(1.0, score)) # Clamp between 0 and 1
                except ValueError:
                    raw_scores[feature] = 0.5

        with open(cache_path, 'w') as f:
            json.dump({
                'public_payload': public_payload,
                'raw_scores': raw_scores,
                'privacy_note': 'Generated only from public task/schema metadata; no private rows or empirical statistics.',
            }, f, indent=2, sort_keys=True)
    raw = np.asarray([float(raw_scores[c]) for c in all_cols], dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError('Public semantic relevance scores must be finite and nonnegative.')
    mean_score = float(raw.mean())
    if mean_score <= 0.0:
        # Unique neutral case: when the scorer assigns no positive relevance,
        # reduce exactly to the unweighted metric. This is not a tuned cutoff.
        relevance = np.ones_like(raw)
    else:
        relevance = raw / mean_score
    weights = {c: float(relevance[i]) for i, c in enumerate(all_cols)}
    print('[PUBLIC SEMANTIC RELEVANCE]', cfg.get('name', cfg.get('task', '')), '| cache:', cache_path, '| scores:', weights)
    return weights


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

def regularize_feature_weights(
    feature_weights,
    shrinkage=0.50,
    min_weight=0.50,
    max_weight=2.00,
):
    """
    Dataset-independent stabilization of mean-normalized feature weights.

    Steps:
    1. Clip extreme weights.
    2. Shrink weights toward the neutral value 1.0.
    3. Renormalize so the mean remains exactly 1.0.

    shrinkage:
        0.0 -> completely uniform weights
        1.0 -> retain clipped Qwen weights
        0.5 -> equal mixture of Qwen and uniform weighting
    """
    if not feature_weights:
        return {}

    names = list(feature_weights.keys())

    values = np.asarray(
        [float(feature_weights[name]) for name in names],
        dtype=float,
    )

    if np.any(~np.isfinite(values)):
        raise ValueError("Feature weights must be finite.")

    # Prevent extremely weak or dominant dimensions.
    values = np.clip(
        values,
        float(min_weight),
        float(max_weight),
    )

    # Shrink toward the neutral unweighted geometry.
    alpha = float(shrinkage)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")

    values = 1.0 + alpha * (values - 1.0)

    # Preserve mean weight = 1.
    mean_value = float(values.mean())

    if mean_value <= 0:
        values = np.ones_like(values)
    else:
        values = values / mean_value

    return {
        name: float(values[i])
        for i, name in enumerate(names)
    }

@torch.inference_mode()
def build_llm_weights_from_dp_demos(cfg, dp_demos, num_cols, cat_cols, schema=None):
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


def _taskaware_weight_vector(num_cols, cat_cols, feature_weights):
    return np.sqrt(np.asarray([max(float(feature_weights.get(c, 1.0)), 0.0) for c in list(num_cols) + list(cat_cols)], dtype=float))

def encode_features_for_dp_kmeans_taskaware(df, num_cols, cat_cols, cat_domains, feature_weights):
    X = encode_features_for_dp_kmeans(df, num_cols, cat_cols, cat_domains)
    wvec = _taskaware_weight_vector(num_cols, cat_cols, feature_weights)
    if X.shape[1] != len(wvec):
        raise ValueError(f'Weight dimension mismatch: X has {X.shape[1]} columns, weights have {len(wvec)}')
    Xw = X * wvec.reshape(1, -1)
    lower = np.zeros(Xw.shape[1], dtype=float)
    upper = np.maximum(wvec, 1e-08)
    return (Xw, lower, upper, wvec)

def encoded_query_vector_taskaware(row_dict, num_cols, cat_cols, cat_domains, feature_weights):
    q = encoded_query_vector(row_dict, num_cols, cat_cols, cat_domains)
    wvec = _taskaware_weight_vector(num_cols, cat_cols, feature_weights)
    if len(q) != len(wvec):
        raise ValueError(f'Query dimension mismatch: q has {len(q)} columns, weights have {len(wvec)}')
    return q * wvec

def unweight_encoded_center(center_w, wvec):
    center_w = np.asarray(center_w, dtype=float)
    wvec = np.asarray(wvec, dtype=float)
    return center_w / np.maximum(wvec, 1e-08)


# -----------------------------------------------------------------------------
# Weighted one-hot geometry for the disjoint DP-prototype LLM method.
# The DP prototypes remain ordinary tabular rows. One-hot encoding is used only
# for DP-KMeans and query-to-center routing on the disjoint main partition.
# -----------------------------------------------------------------------------

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

def mixed_dist_taskaware(row_dict, cand, num_cols, cat_cols, feature_weights):
    d_num = 0.0
    for c in num_cols:
        w = float(feature_weights.get(c, 1.0))
        d_num += w * abs(float(row_dict[c]) - float(cand[c]))
    d_cat = 0.0
    for c in cat_cols:
        w = float(feature_weights.get(c, 1.0))
        d_cat += w * (1.0 if str(row_dict[c]) != str(cand[c]) else 0.0)
    return d_num + d_cat

def retrieve_balanced_nearest_shots_taskaware(pool_df, row_dict, k, num_cols, cat_cols, feature_weights, seed=None, label_col='target'):
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
        out['_dist'] = out.apply(lambda r: mixed_dist_taskaware(row_dict, {c: r[c] for c in num_cols + cat_cols}, num_cols, cat_cols, feature_weights), axis=1)
        return out
    pos = _add_dist(pos)
    neg = _add_dist(neg)
    k_pos = k // 2
    k_neg = k - k_pos
    pos_shots = pos.sort_values('_dist').head(k_pos) if len(pos) > 0 else pos
    neg_shots = neg.sort_values('_dist').head(k_neg) if len(neg) > 0 else neg
    shots = pd.concat([pos_shots, neg_shots], axis=0)
    remaining = k - len(shots)
    if remaining > 0:
        used_idx = set(shots.index.tolist())
        leftover = pd.concat([pos, neg], axis=0)
        leftover = leftover[~leftover.index.isin(used_idx)].sort_values('_dist')
        shots = pd.concat([shots, leftover.head(remaining)], axis=0)
    if seed is not None and len(shots) > 0:
        shots = shots.sample(frac=1.0, random_state=seed)
    return shots.drop(columns=['_dist'], errors='ignore').reset_index(drop=True)

def strat_A2_clean_fewshot(cfg, row_dict, pool_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(pool_df))
    shots = retrieve_balanced_nearest_shots(pool_df, row_dict, k, num_cols, cat_cols, seed=seed) if k > 0 else None
    if shots is not None:
        print('[DEBUG A2] shot label counts:', shots['target'].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, num_cols + cat_cols)

def fit_diffprivlib_dp_kmeans(df, num_cols, cat_cols, cat_domains, k, eps_cluster, random_state):
    if not HAS_DIFFPRIVLIB_KMEANS:
        raise ImportError(f'diffprivlib.models.KMeans is not available. Original import error: {DIFFPRIVLIB_IMPORT_ERROR}')
    X = encode_features_for_dp_kmeans(df, num_cols, cat_cols, cat_domains)
    n, d = X.shape
    if n == 0:
        raise ValueError('Cannot fit DP-KMeans on empty dataframe.')
    k_eff = int(min(max(1, k), n))
    lower = np.zeros(d)
    upper = np.ones(d)
    bounds = (lower, upper)
    model = DPKMeans(n_clusters=k_eff, epsilon=float(eps_cluster), bounds=bounds, random_state=int(random_state))
    model.fit(X)
    centers = np.asarray(model.cluster_centers_, dtype=float)
    labels = np.asarray(model.labels_, dtype=int)
    return (centers, labels)

def fit_diffprivlib_dp_kmeans_taskaware(df, num_cols, cat_cols, cat_domains, feature_weights, k, eps_cluster, random_state):
    if not HAS_DIFFPRIVLIB_KMEANS:
        raise ImportError(f'diffprivlib.models.KMeans is not available. Original import error: {DIFFPRIVLIB_IMPORT_ERROR}')
    Xw, lower, upper, wvec = encode_features_for_dp_kmeans_taskaware(df, num_cols, cat_cols, cat_domains, feature_weights)
    n, d = Xw.shape
    if n == 0:
        raise ValueError('Cannot fit task-aware DP-KMeans on empty dataframe.')
    k_eff = int(min(max(1, k), n))
    model = DPKMeans(n_clusters=k_eff, epsilon=float(eps_cluster), bounds=(lower, upper), random_state=int(random_state))
    model.fit(Xw)
    centers_w = np.asarray(model.cluster_centers_, dtype=float)
    labels = np.asarray(model.labels_, dtype=int)
    return (centers_w, labels, wvec)

def build_C2_global_aim_pool(train_df, num_cols, cat_cols, eps_total, n_synth=None, n_bins=8):
    df = train_df.copy()
    if n_synth is None:
        n_synth = len(df)
    df = clip_numeric_df(df, num_cols, 0.0, 1.0)
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    df = df[ordered_cols].copy()
    bin_edges = _fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
    df_aim = _apply_numeric_binning(df, num_cols, bin_edges)
    for c in cat_cols:
        df_aim[c] = df_aim[c].astype(str)
    df_aim['target'] = df_aim['target'].astype(str)
    synth = Synthesizer.create('aim', epsilon=float(eps_total), verbose=False)
    synth.fit(df_aim, preprocessor_eps=0.0)
    synth_df = synth.sample(int(n_synth)).copy()
    synth_df = _decode_binned_numeric(synth_df, num_cols, bin_edges)
    for c in cat_cols:
        synth_df[c] = synth_df[c].astype(str)
    synth_df['target'] = pd.to_numeric(synth_df['target'], errors='coerce').fillna(0).astype(int).clip(0, 1)
    return synth_df[ordered_cols].reset_index(drop=True)

def build_C3_global_privsyn_pool(train_df, num_cols, cat_cols, cat_domains, eps_total, n_synth=None, n_bins=8, delta=1e-05, privsyn_home=None, consistency_iterations=2, view_iterations=10, gum_iterations=20, label_target_only=False):
    import os
    import sys
    import copy
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    if n_synth is None:
        n_synth = len(train_df)
    if float(eps_total) <= 0:
        raise ValueError('eps_total must be positive for PrivSyn.')
    candidate_homes = []
    if privsyn_home is not None:
        candidate_homes.append(os.path.expanduser(str(privsyn_home)))
    if os.environ.get('PRIVSYN_HOME'):
        candidate_homes.append(os.path.expanduser(os.environ['PRIVSYN_HOME']))
    candidate_homes.extend([os.path.expanduser('~/PrivSyn'), os.path.join(os.getcwd(), 'PrivSyn')])
    privsyn_root = None
    for cand in candidate_homes:
        if cand and os.path.exists(os.path.join(cand, 'Anonymisation.py')) and os.path.exists(os.path.join(cand, 'GUM.py')):
            privsyn_root = cand
            break
    if privsyn_root is None:
        raise ImportError('Official/external PrivSyn code not found. Clone it and set PRIVSYN_HOME, e.g.:\n  git clone https://github.com/sukjingitsit/PrivSyn.git ~/PrivSyn\n  export PRIVSYN_HOME=~/PrivSyn\n')
    if privsyn_root not in sys.path:
        sys.path.insert(0, privsyn_root)
    try:
        from Anonymisation import Anonymisation
        from Consistenter import Consistenter
        from GUM import GraduallyUpdateMethod
    except Exception as e:
        raise ImportError(f'Found PrivSyn at {privsyn_root}, but could not import its modules. Make sure dependencies are installed: pyyaml numpy pandas networkx scikit-learn.\nOriginal error: {repr(e)}')
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    df = train_df[ordered_cols].copy().reset_index(drop=True)
    df = clip_numeric_df(df, num_cols, lo=0.0, hi=1.0)
    bin_edges = _fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
    encoded = pd.DataFrame(index=df.index)
    encode_mapping = {}
    decode_mapping = {}
    encode_schema = {}
    for c in num_cols:
        vals = pd.to_numeric(df[c], errors='coerce').fillna(0.0).clip(0.0, 1.0).to_numpy()
        idx = np.digitize(vals, bin_edges[c][1:-1], right=False).astype(int)
        encoded[c] = idx
        domain = list(range(n_bins))
        encode_schema[c] = domain
        encode_mapping[c] = {i: i for i in domain}
        decode_mapping[c] = domain
    for c in cat_cols:
        domain_vals = list(map(str, cat_domains.get(c, [])))
        if len(domain_vals) == 0:
            raise ValueError(f'Missing public categorical domain for PrivSyn column: {c}')
        val_to_idx = {v: i for i, v in enumerate(domain_vals)}
        encoded[c] = df[c].astype(str).map(val_to_idx).fillna(0).astype(int)
        domain = list(range(len(domain_vals)))
        encode_schema[c] = domain
        encode_mapping[c] = val_to_idx
        decode_mapping[c] = domain_vals
    encoded['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int).clip(0, 1)
    encode_schema['target'] = [0, 1]
    encode_mapping['target'] = {0: 0, 1: 1}
    decode_mapping['target'] = [0, 1]
    loader = SimpleNamespace()
    loader.private_data = encoded[ordered_cols].copy()
    loader.all_attrs = ordered_cols
    loader.encode_mapping = encode_mapping
    loader.decode_mapping = decode_mapping
    loader.encode_schema = encode_schema
    loader.priv_marginals = {}
    loader.priv_one_way = {}
    loader.priv_two_way = {}
    loader.priv_indif = {}

    def _one_way_marginal(attribute, data):
        marginal = data.assign(n=1).pivot_table(values='n', index=attribute, aggfunc='sum', fill_value=0)
        indices = sorted(loader.encode_schema[attribute])
        marginal = marginal.reindex(index=indices).fillna(0).astype(np.int32)
        loader.priv_marginals[frozenset([attribute])] = marginal
        loader.priv_one_way[frozenset([attribute])] = marginal
        return marginal

    def _two_way_marginal(a, b, data):
        marginal = data.assign(n=1).pivot_table(values='n', index=[a, b], aggfunc='sum', fill_value=0)
        ia = sorted(loader.encode_schema[a])
        ib = sorted(loader.encode_schema[b])
        marginal = marginal.reindex(pd.MultiIndex.from_product([ia, ib], names=[a, b])).fillna(0).astype(np.int32)
        loader.priv_marginals[frozenset([a, b])] = marginal
        loader.priv_two_way[frozenset([a, b])] = marginal
        return marginal

    def _indif(a, b, data):
        n = max(len(data), 1)
        ma = _one_way_marginal(a, data)
        mb = _one_way_marginal(b, data)
        mab = _two_way_marginal(a, b, data)
        ia = sorted(loader.encode_schema[a])
        ib = sorted(loader.encode_schema[b])
        val = 0.0
        for xa in ia:
            for xb in ib:
                expected = float(ma.loc[xa, 'n']) * float(mb.loc[xb, 'n']) / float(n)
                actual = float(mab.loc[(xa, xb), 'n'])
                val += abs(expected - actual)
        loader.priv_indif[frozenset([a, b])] = val
        return val
    for c in ordered_cols:
        _one_way_marginal(c, loader.private_data)
    for i, a in enumerate(ordered_cols):
        for b in ordered_cols[i + 1:]:
            _indif(a, b, loader.private_data)
            if label_target_only:
                if a != 'target' and b != 'target':
                    loader.priv_indif[frozenset([a, b])] = 0.0
    print('[OFFICIAL PRIVSYN]', 'root:', privsyn_root, '| eps:', eps_total, '| delta:', delta, '| n_synth:', n_synth, '| n_bins:', n_bins, '| attrs:', len(ordered_cols))
    anonymiser = Anonymisation(float(eps_total), float(delta))
    anonymiser.anonymiser(loader)
    consistenter = Consistenter(anonymiser, loader.all_attrs)
    consistenter.make_consistent(iterations=int(consistency_iterations))
    gum = GraduallyUpdateMethod(loader, consistenter)
    gum.initialiser(view_iterations=int(view_iterations))
    synth_encoded = gum.synthesize(iterations=int(gum_iterations), num_records=int(n_synth)).copy()
    synth_df = pd.DataFrame(index=synth_encoded.index)
    for c in num_cols:
        ids = pd.to_numeric(synth_encoded[c], errors='coerce').fillna(0).astype(int).clip(0, n_bins - 1)
        mids = (bin_edges[c][:-1] + bin_edges[c][1:]) / 2.0
        synth_df[c] = mids[ids.to_numpy()]
    for c in cat_cols:
        domain_vals = list(map(str, cat_domains[c]))
        ids = pd.to_numeric(synth_encoded[c], errors='coerce').fillna(0).astype(int).clip(0, len(domain_vals) - 1)
        synth_df[c] = [domain_vals[i] for i in ids.to_numpy()]
    synth_df['target'] = pd.to_numeric(synth_encoded['target'], errors='coerce').fillna(0).astype(int).clip(0, 1)
    synth_df = synth_df[ordered_cols].reset_index(drop=True)
    print('[DEBUG C3_GLOBAL_PRIVSYN official]', 'n_synth:', len(synth_df), 'labels:', synth_df['target'].value_counts().to_dict())
    return synth_df

def build_C3_privsyn_ta_construct(train_df, num_cols, cat_cols, cat_domains, eps_cluster, eps_synth, k, rng, rows_per_cluster, feature_weights, random_state, n_bins=8, delta=1e-05, privsyn_home=None, consistency_iterations=2, view_iterations=10, gum_iterations=20, min_cluster_size=5):
    eps_cluster = float(eps_cluster)
    eps_synth = float(eps_synth)
    if eps_cluster <= 0 or eps_synth <= 0:
        raise ValueError('eps_cluster and eps_synth must be positive for C3_PRIVSYN_TA_CONSTRUCT.')
    centers_w, clusters, wvec = fit_diffprivlib_dp_kmeans_taskaware(train_df, num_cols, cat_cols, cat_domains, feature_weights=feature_weights, k=k, eps_cluster=eps_cluster, random_state=random_state)
    rows_per_cluster = int(max(1, rows_per_cluster))
    synth_store = {}
    centers_out = []
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    for c in range(int(k)):
        if c < len(centers_w):
            center_w = centers_w[c]
            center_unweighted = unweight_encoded_center(center_w, wvec)
            df_c = train_df[clusters == c].copy().reset_index(drop=True)
        else:
            center_w = _public_encoded_center(num_cols, cat_cols) * wvec
            center_unweighted = _public_encoded_center(num_cols, cat_cols)
            df_c = train_df.iloc[0:0].copy().reset_index(drop=True)
        centers_out.append(center_w)
        if DP_PRIVATE_DEBUG:
            print(f"[DEBUG C3_PRIVSYN_TA_CONSTRUCT real] cluster={c} size={len(df_c)} labels={(df_c['target'].value_counts().to_dict() if len(df_c) > 0 else {})}")
        if len(df_c) < int(min_cluster_size):
            fallback_rows = make_c2adp_centroid_fallback(center=center_unweighted, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, n_rows=rows_per_cluster)
            synth_store[c] = fallback_rows[ordered_cols].reset_index(drop=True)
            print(f'[DEBUG C3_PRIVSYN_TA_CONSTRUCT fallback] cluster={c} real_size={len(df_c)} synth_size={len(fallback_rows)}')
            continue
        set_seed(int(rng.randint(0, 10 ** 9)))
        synth_c = build_C3_global_privsyn_pool(train_df=df_c[ordered_cols].copy(), num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, eps_total=eps_synth, n_synth=rows_per_cluster, n_bins=n_bins, delta=delta, privsyn_home=privsyn_home, consistency_iterations=consistency_iterations, view_iterations=view_iterations, gum_iterations=gum_iterations)
        synth_store[c] = synth_c[ordered_cols].reset_index(drop=True)
        print(f"[DEBUG C3_PRIVSYN_TA_CONSTRUCT synth] cluster={c} synth_size={len(synth_store[c])} synth_labels={synth_store[c]['target'].value_counts().to_dict()}")
    return (np.asarray(centers_out, dtype=float), synth_store)

def _eps_delta_to_rho(epsilon, delta):
    epsilon = max(float(epsilon), 0.0)
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError('delta must be in (0,1) for epsilon-to-rho conversion.')
    log_term = np.log(1.0 / delta)
    return float((np.sqrt(log_term + epsilon) - np.sqrt(log_term)) ** 2)

def noise_aware_star_graph_selection(feature_scores, eps_synth, delta=1e-05, public_reference_n=1000):
    """Select the best nonempty prefix using PrivSyn's fixed publication share."""
    if not feature_scores:
        return ([], {'selected_r': 0, 'objective_by_r': {}})
    rho_synth = _eps_delta_to_rho(eps_synth, delta)
    rho_measure = max(PRIVSYN_MARGINAL_PUBLICATION_RHO_SHARE * rho_synth, 1e-12)
    n_ref = max(int(public_reference_n), 1)
    ranked_features = sorted(
        feature_scores,
        key=lambda feature: max(0.0, float(feature_scores[feature].get('icl_score', 0.0))),
        reverse=True,
    )
    objective_by_r = {}
    cumulative_benefit = 0.0
    cumulative_q = 0.0
    for r, feature in enumerate(ranked_features, start=1):
        score = float(feature_scores[feature].get('icl_score', 0.0))
        if not np.isfinite(score):
            score = 0.0
        cumulative_benefit += max(0.0, score)
        q_cells = int(feature_scores[feature].get('pair_cell_count', 2))
        cumulative_q += max(q_cells, 1)
        expected_l1_noise = cumulative_q / float(n_ref) * np.sqrt(float(r) / (np.pi * rho_measure))
        objective_by_r[r] = {
            'benefit': float(cumulative_benefit),
            'expected_l1_noise': float(expected_l1_noise),
            'objective': float(cumulative_benefit - expected_l1_noise),
        }
    best_r = max(objective_by_r, key=lambda r: objective_by_r[r]['objective'])
    selected = ranked_features[:best_r]
    diagnostics = {
        'selected_r': int(best_r),
        'ranked_features': ranked_features,
        'rho_synth': float(rho_synth),
        'rho_measure': float(rho_measure),
        'public_reference_n': int(n_ref),
        'privsyn_publication_share': float(PRIVSYN_MARGINAL_PUBLICATION_RHO_SHARE),
        'objective_by_r': objective_by_r,
    }
    return (selected, diagnostics)


def build_C3_privsyn_icl_all_star(train_df, num_cols, cat_cols, cat_domains,
                                  eps_cluster, eps_synth, k, rng,
                                  rows_per_cluster, feature_weights,
                                  random_state, n_bins=8, delta=1e-05,
                                  privsyn_home=None, consistency_iterations=2,
                                  view_iterations=10, gum_iterations=20):
    """Disjoint DP-prototype LLM weights + weighted one-hot DP clustering.

    DP prototypes are created before this function and remain ordinary tabular
    rows. This function applies their LLM-derived feature weights to a one-hot
    representation of the disjoint main partition, then runs local PrivSyn.
    """
    centers_w, clusters, onehot_spec = fit_diffprivlib_dp_kmeans_onehot_weighted(
        df=train_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_domains=cat_domains,
        feature_weights=feature_weights,
        k=k,
        eps_cluster=float(eps_cluster),
        random_state=random_state,
    )
    all_features = list(num_cols) + list(cat_cols)
    ordered_cols = all_features + ['target']
    synth_store, centers_out, log = {}, [], {}
    rows_per_cluster = int(max(1, rows_per_cluster))
    for c in range(int(k)):
        if c < len(centers_w):
            center_w = centers_w[c]
            df_c = train_df[clusters == c].copy().reset_index(drop=True)
        else:
            center_w = np.zeros(int(onehot_spec['dimension']), dtype=float)
            df_c = train_df.iloc[0:0].copy().reset_index(drop=True)
        centers_out.append(center_w)
        center_decoded = decode_weighted_onehot_center(center_w, onehot_spec)
        fallback = make_c2adp_centroid_fallback(
            center_decoded, num_cols, cat_cols, cat_domains, rows_per_cluster
        )
        if len(df_c) == 0:
            synth_store[c] = fallback[ordered_cols].reset_index(drop=True)
            log[c] = {
                'used_features': all_features,
                'selection_rule': 'all_label_feature_marginals_fixed_workload',
                'encoding': 'weighted_onehot',
                'fallback': True,
            }
            continue
        try:
            set_seed(int(rng.randint(0, 10 ** 9)))
            synth = build_C3_global_privsyn_pool(
                train_df=df_c[ordered_cols],
                num_cols=num_cols,
                cat_cols=cat_cols,
                cat_domains=cat_domains,
                eps_total=float(eps_synth),
                n_synth=rows_per_cluster,
                n_bins=n_bins,
                delta=delta,
                privsyn_home=privsyn_home,
                consistency_iterations=consistency_iterations,
                view_iterations=view_iterations,
                gum_iterations=gum_iterations,
                label_target_only=True,
            )
            synth_store[c] = synth[ordered_cols].reset_index(drop=True)
            log[c] = {
                'used_features': all_features,
                'selection_rule': 'all_label_feature_marginals_fixed_workload',
                'encoding': 'weighted_onehot',
                'fallback': False,
                'private_cluster_rows': int(len(df_c)),
            }
        except Exception as exc:
            print(f'[DP-SAMPLE LLM FALLBACK] cluster={c} error={exc!r}')
            synth_store[c] = fallback[ordered_cols].reset_index(drop=True)
            log[c] = {
                'used_features': all_features,
                'selection_rule': 'all_label_feature_marginals_fixed_workload',
                'encoding': 'weighted_onehot',
                'fallback': True,
                'private_cluster_rows': int(len(df_c)),
                'error': repr(exc),
            }
    return np.asarray(centers_out, dtype=float), synth_store, log, onehot_spec


def build_C3_privsyn_icl_construct(train_df, num_cols, cat_cols, cat_domains, eps_cluster, eps_synth, k, rng, rows_per_cluster, feature_weights, random_state, n_bins=8, delta=1e-05, privsyn_home=None, consistency_iterations=2, view_iterations=10, gum_iterations=20):
    eps_cluster = float(eps_cluster)
    eps_synth = float(eps_synth)
    if eps_cluster <= 0 or eps_synth <= 0:
        raise ValueError('eps_cluster and eps_synth must be positive.')
    all_features = list(num_cols) + list(cat_cols)
    ordered_cols = all_features + ['target']
    centers_w, clusters, wvec = fit_diffprivlib_dp_kmeans_taskaware(train_df, num_cols, cat_cols, cat_domains, feature_weights=feature_weights, k=k, eps_cluster=eps_cluster, random_state=random_state)
    rows_per_cluster = int(max(1, rows_per_cluster))
    synth_store = {}
    centers_out = []
    selection_log = {}
    for c in range(int(k)):
        if c < len(centers_w):
            center_w = centers_w[c]
            center_unweighted = unweight_encoded_center(center_w, wvec)
            df_c = train_df[clusters == c].copy().reset_index(drop=True)
        else:
            center_w = _public_encoded_center(num_cols, cat_cols) * wvec
            center_unweighted = _public_encoded_center(num_cols, cat_cols)
            df_c = train_df.iloc[0:0].copy().reset_index(drop=True)
        centers_out.append(center_w)
        fallback_rows = make_c2adp_centroid_fallback(center=center_unweighted, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, n_rows=rows_per_cluster)
        if len(df_c) == 0:
            synth_store[c] = fallback_rows[ordered_cols].reset_index(drop=True)
            selection_log[c] = {'used_features': [], 'total_features_evaluated': len(all_features), 'selection_rule': 'dp_centroid_fallback_empty_region', 'selection_diagnostics': {}}
            print(f'[C3 EMPTY REGION] cluster={c} fallback=dp_centroid')
            continue
        scores = {}

        for feature in all_features:
            # feature_weights stores mean-normalized public semantic relevance.
            normalized_relevance = max(float(feature_weights.get(feature, 1.0)), 0.0)
            task_weight = normalized_relevance

            if feature in num_cols:
                domain_size = int(n_bins)
            else:
                domain_size = int(
                    len(cat_domains[feature])
                )

            pair_cell_count = int(
                domain_size * 2
            )

            domain_complexity = np.sqrt(
                float(pair_cell_count)
            )

            icl_score = (
                normalized_relevance
                / max(domain_complexity, 1.0)
            )

            scores[feature] = {
                "task_weight": task_weight,
                "normalized_relevance": float(
                    normalized_relevance
                ),
                "icl_score": float(icl_score),
                "feature_domain_size": domain_size,
                "pair_cell_count": pair_cell_count,
            }

        selected_features, selection_diagnostics = (
            noise_aware_star_graph_selection(
                feature_scores=scores,
                eps_synth=eps_synth,
                delta=delta,
                public_reference_n=rows_per_cluster,
            )
        )
        best_r = selection_diagnostics['selected_r']
        best_info = selection_diagnostics['objective_by_r'][best_r]
        print(f"[NOISE-AWARE SELECT] cluster={c} eps_synth={eps_synth:.6f} rho_synth={selection_diagnostics['rho_synth']:.8f} rho_measure={selection_diagnostics['rho_measure']:.8f} selected_r={best_r} benefit={best_info['benefit']:.6f} expected_l1_noise={best_info['expected_l1_noise']:.6f} objective={best_info['objective']:.6f}")
        local_cols = selected_features + ['target']
        df_local = df_c[local_cols].copy()
        local_num_cols = [col for col in num_cols if col in selected_features]
        local_cat_cols = [col for col in cat_cols if col in selected_features]
        try:
            set_seed(int(rng.randint(0, 10 ** 9)))
            synth_local = build_C3_global_privsyn_pool(train_df=df_local, num_cols=local_num_cols, cat_cols=local_cat_cols, cat_domains=cat_domains, eps_total=eps_synth, delta=delta, n_synth=rows_per_cluster, n_bins=n_bins, privsyn_home=privsyn_home, consistency_iterations=consistency_iterations, view_iterations=view_iterations, gum_iterations=gum_iterations, label_target_only=True)
            for col in ordered_cols:
                if col not in synth_local.columns:
                    synth_local[col] = fallback_rows[col].values[:len(synth_local)]
            synth_local = synth_local[ordered_cols].copy().reset_index(drop=True)
        except Exception as exc:
            print(f'[C3 PRIVSYN FALLBACK] cluster={c} error={exc!r}')
            synth_local = fallback_rows[ordered_cols].reset_index(drop=True)
        synth_store[c] = synth_local
        selection_log[c] = {'used_features': selected_features, 'total_features_evaluated': len(all_features), 'selection_rule': 'public_llm_semantic_relevance_noise_aware', 'selection_diagnostics': selection_diagnostics}
        print(f'[C3 BUILD] cluster={c} features_kept={len(selected_features)}/{len(all_features)} synth_size={len(synth_local)}')
    return (np.asarray(centers_out, dtype=float), synth_store, selection_log)

def build_C2aDP_taskaware_kmeans_mst(train_df, num_cols, cat_cols, eps_cluster, eps_synth, k, rng, rows_per_cluster, cat_domains, feature_weights, random_state, min_cluster_size=5):
    centers_w, clusters, wvec = fit_diffprivlib_dp_kmeans_taskaware(train_df, num_cols, cat_cols, cat_domains, feature_weights=feature_weights, k=k, eps_cluster=eps_cluster, random_state=random_state)
    rows_per_cluster = int(max(1, rows_per_cluster))
    synth_store = {}
    centers_out = []
    for c in range(int(k)):
        if c < len(centers_w):
            center_w = centers_w[c]
            center_unweighted = unweight_encoded_center(center_w, wvec)
            df_c = train_df[clusters == c].copy()
        else:
            center_w = _public_encoded_center(num_cols, cat_cols) * wvec
            center_unweighted = _public_encoded_center(num_cols, cat_cols)
            df_c = train_df.iloc[0:0].copy()
        centers_out.append(center_w)
        print(f"[DEBUG C2aDP_TA real] cluster={c} size={len(df_c)} labels={(df_c['target'].value_counts().to_dict() if len(df_c) > 0 else {})}")
        if len(df_c) == 0:
            fallback_rows = make_c2adp_centroid_fallback(center=center_unweighted, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, n_rows=rows_per_cluster)
            synth_store[c] = fallback_rows.reset_index(drop=True)
            print(f'[DEBUG C2aDP_TA fallback] cluster={c} real_size={len(df_c)} synth_size={len(fallback_rows)}')
            continue
        set_seed(int(rng.randint(0, 10 ** 9)))
        syn_df = mst_sample_cluster(df_c, num_cols, cat_cols, eps_synth, rows_per_cluster)
        synth_store[c] = syn_df.reset_index(drop=True)
        print(f"[DEBUG C2aDP_TA synth] cluster={c} synth_size={len(synth_store[c])} synth_labels={synth_store[c]['target'].value_counts().to_dict()}")
    return (np.asarray(centers_out, dtype=float), synth_store)

def build_C2aDP_ta_construct(train_df, num_cols, cat_cols, eps_cluster, eps_synth, k, rng, rows_per_cluster, cat_domains, feature_weights, random_state, min_cluster_size=5, ta_mst_top_m=3):
    centers_w, clusters, wvec = fit_diffprivlib_dp_kmeans_taskaware(train_df, num_cols, cat_cols, cat_domains, feature_weights=feature_weights, k=k, eps_cluster=eps_cluster, random_state=random_state)
    rows_per_cluster = int(max(1, rows_per_cluster))
    synth_store = {}
    centers_out = []
    for c in range(int(k)):
        if c < len(centers_w):
            center_w = centers_w[c]
            center_unweighted = unweight_encoded_center(center_w, wvec)
            df_c = train_df[clusters == c].copy()
        else:
            center_w = _public_encoded_center(num_cols, cat_cols) * wvec
            center_unweighted = _public_encoded_center(num_cols, cat_cols)
            df_c = train_df.iloc[0:0].copy()
        centers_out.append(center_w)
        print(f"[DEBUG C2aDP_TA_CONSTRUCT real] cluster={c} size={len(df_c)} labels={(df_c['target'].value_counts().to_dict() if len(df_c) > 0 else {})}")
        if len(df_c) == 0:
            fallback_rows = make_c2adp_centroid_fallback(center=center_unweighted, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, n_rows=rows_per_cluster)
            synth_store[c] = fallback_rows.reset_index(drop=True)
            print(f'[DEBUG C2aDP_TA_CONSTRUCT fallback] cluster={c} synth_size={len(fallback_rows)}')
            continue
        set_seed(int(rng.randint(0, 10 ** 9)))
        syn_df = task_oriented_pgm_sample_cluster(df_cluster=df_c, num_cols=num_cols, cat_cols=cat_cols, eps_cluster=eps_synth, n_rows=rows_per_cluster, feature_weights=feature_weights, cat_domains=cat_domains, n_bins=8, top_m=ta_mst_top_m)
        synth_store[c] = syn_df.reset_index(drop=True)
        print(f"[DEBUG C2aDP_TA_CONSTRUCT synth] cluster={c} synth_size={len(synth_store[c])} synth_labels={synth_store[c]['target'].value_counts().to_dict()}")
    return (np.asarray(centers_out, dtype=float), synth_store)

def _public_encoded_center(num_cols, cat_cols):
    d = len(num_cols) + len(cat_cols)
    return np.full(d, 0.5, dtype=float)

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

def build_C2aDP_diffprivlib_kmeans_mst(train_df, num_cols, cat_cols, eps_cluster, eps_synth, k, rng, rows_per_cluster, cat_domains, random_state, min_cluster_size=5):
    centers, clusters = fit_diffprivlib_dp_kmeans(train_df, num_cols, cat_cols, cat_domains, k=k, eps_cluster=eps_cluster, random_state=random_state)
    rows_per_cluster = int(max(1, rows_per_cluster))
    synth_store = {}
    centers_out = []
    for c in range(int(k)):
        if c < len(centers):
            center_c = centers[c]
            df_c = train_df[clusters == c].copy()
        else:
            center_c = _public_encoded_center(num_cols, cat_cols)
            df_c = train_df.iloc[0:0].copy()
        centers_out.append(center_c)
        print(f"[DEBUG C2aDP real] cluster={c} size={len(df_c)} labels={(df_c['target'].value_counts().to_dict() if len(df_c) > 0 else {})}")
        if len(df_c) == 0:
            fallback_rows = make_c2adp_centroid_fallback(center=center_c, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, n_rows=rows_per_cluster)
            synth_store[c] = fallback_rows.reset_index(drop=True)
            print(f'[DEBUG C2aDP fallback] cluster={c} real_size={len(df_c)} synth_size={len(fallback_rows)}')
            continue
        set_seed(int(rng.randint(0, 10 ** 9)))
        syn_df = mst_sample_cluster(df_c, num_cols, cat_cols, eps_synth, rows_per_cluster)
        synth_store[c] = syn_df.reset_index(drop=True)
        print(f"[DEBUG C2aDP synth] cluster={c} synth_size={len(synth_store[c])} synth_labels={synth_store[c]['target'].value_counts().to_dict()}")
    return (np.asarray(centers_out, dtype=float), synth_store)

def build_B1_ldp_pool(train_df, num_cols, cat_cols, eps_total, rng, cat_domains):
    e_col = eps_total / max(len(num_cols) + len(cat_cols) + 1, 1)
    df_dp = train_df.copy()
    for c in cat_cols:
        df_dp[c] = df_dp[c].astype(str)
    if len(num_cols) > 0:
        noisy = df_dp[num_cols] + rng.laplace(0, 1.0 / e_col, size=df_dp[num_cols].shape)
        df_dp[num_cols] = noisy.clip(0.0, 1.0)
    for c in cat_cols:
        df_dp[c] = df_dp[c].apply(lambda x: rr(rng, str(x), cat_domains[c], e_col))
    df_dp['target'] = df_dp['target'].apply(lambda x: rr(rng, int(x), [0, 1], e_col))
    return df_dp

def strat_B1_ldp(cfg, row_dict, dp_pool_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(dp_pool_df))
    shots = retrieve_nearest_shots(dp_pool_df, row_dict, k, num_cols, cat_cols, seed=seed) if k > 0 else None
    if shots is not None:
        print('[DEBUG B1] shot label counts:', shots['target'].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, num_cols + cat_cols)

def build_C1_mst_pool(train_df, num_cols, cat_cols, eps_total, n_synth=None, n_bins=8):
    df = train_df.copy()
    if n_synth is None:
        n_synth = len(df)
    df = clip_numeric_df(df, num_cols, 0.0, 1.0)
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    df = df[ordered_cols].copy()
    bin_edges = _fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
    df_mst = _apply_numeric_binning(df, num_cols, bin_edges)
    for c in cat_cols:
        df_mst[c] = df_mst[c].astype(str)
    df_mst['target'] = df_mst['target'].astype(str)
    synth = Synthesizer.create('mst', epsilon=float(eps_total), verbose=False)
    synth.fit(df_mst, preprocessor_eps=0.0)
    synth_df = synth.sample(n_synth).copy()
    synth_df = _decode_binned_numeric(synth_df, num_cols, bin_edges)
    for c in cat_cols:
        synth_df[c] = synth_df[c].astype(df[c].dtype if c in df.columns else str)
    synth_df['target'] = pd.to_numeric(synth_df['target'], errors='coerce').fillna(0).astype(int)
    return synth_df[ordered_cols].reset_index(drop=True)

def select_task_mst_features(feature_weights, num_cols, cat_cols, cat_domains, top_m=None, max_cat_domain=16):
    selected = []
    for c in num_cols:
        selected.append(c)
    for c in cat_cols:
        domain = list(map(str, cat_domains.get(c, [])))
        dom_size = len(domain)
        if max_cat_domain is not None and dom_size > int(max_cat_domain):
            print(f'[TASK-GUIDED MST skip high-cardinality cat] {c} domain_size={dom_size} max_cat_domain={max_cat_domain}')
            continue
        selected.append(c)
    print('[TASK-GUIDED MST Option-A all eligible features]', selected)
    return selected

def task_oriented_pgm_sample_cluster(df_cluster, num_cols, cat_cols, eps_cluster, n_rows, feature_weights, cat_domains, n_bins=8, top_m=None):
    df = df_cluster.copy()
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    if len(df) == 0:
        return pd.DataFrame(columns=ordered_cols)
    df = clip_numeric_df(df, num_cols, 0.0, 1.0)
    df = df[ordered_cols].copy()
    bin_edges = _fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
    df_discrete = _apply_numeric_binning(df, num_cols, bin_edges)
    for c in num_cols:
        df_discrete[c] = df_discrete[c].astype(int)
    for c in cat_cols:
        domain_vals = list(map(str, cat_domains.get(c, [])))
        if not domain_vals:
            df_discrete[c] = 0
            continue
        val_to_idx = {val: i for i, val in enumerate(domain_vals)}
        df_discrete[c] = df_discrete[c].astype(str).map(val_to_idx).fillna(0).astype(int)
    df_discrete['target'] = df_discrete['target'].astype(int)
    domain_dict = {}
    for c in num_cols:
        domain_dict[c] = n_bins
    for c in cat_cols:
        domain_dict[c] = max(len(cat_domains.get(c, [])), 1)
    domain_dict['target'] = 2
    domain = Domain(domain_dict.keys(), domain_dict.values())
    data = Dataset(df_discrete, domain)
    selected_features = select_task_mst_features(feature_weights=feature_weights, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, top_m=top_m, max_cat_domain=16)
    cliques = [('target',)]
    for f in selected_features:
        if f in domain.attrs:
            cliques.append((f, 'target'))
    print('[STAR-PGM cliques]', cliques)
    eps_per_clique = float(eps_cluster) / max(len(cliques), 1)
    if eps_per_clique <= 0:
        raise ValueError('eps_per_clique must be positive')
    laplace_scale = 1.0 / eps_per_clique
    sigma = np.sqrt(2.0) * laplace_scale
    measurements = []
    for clq in cliques:
        true_vector = data.project(clq).datavector()
        noisy_vector = true_vector + np.random.laplace(loc=0.0, scale=laplace_scale, size=true_vector.shape)
        I = np.eye(len(true_vector))
        measurements.append((I, noisy_vector, sigma, clq))
    engine = FactoredInference(domain, log=False, iters=1000)
    model = engine.estimate(measurements, total=len(df))
    synth_df = model.synthetic_data(rows=int(n_rows)).df
    for c in num_cols:
        if c not in synth_df.columns:
            synth_df[c] = 0
    for c in cat_cols:
        if c not in synth_df.columns:
            domain_vals = list(map(str, cat_domains.get(c, [])))
            synth_df[c] = domain_vals[0] if len(domain_vals) > 0 else 'NA'
        else:
            domain_vals = list(map(str, cat_domains.get(c, [])))
            if not domain_vals:
                synth_df[c] = 'NA'
            else:
                idx_to_val = {i: val for i, val in enumerate(domain_vals)}
                synth_df[c] = synth_df[c].map(idx_to_val).fillna(domain_vals[0])
    if 'target' not in synth_df.columns:
        synth_df['target'] = 0
    synth_df = _decode_binned_numeric(synth_df, num_cols, bin_edges)
    synth_df['target'] = pd.to_numeric(synth_df['target'], errors='coerce').fillna(0).astype(int)
    return synth_df[ordered_cols].reset_index(drop=True)

def mst_sample_cluster(df_cluster, num_cols, cat_cols, eps_cluster, n_rows, n_bins=8):
    df = df_cluster.copy()
    if len(df) == 0:
        return pd.DataFrame(columns=list(num_cols) + list(cat_cols) + ['target'])
    df = clip_numeric_df(df, num_cols, 0.0, 1.0)
    ordered_cols = list(num_cols) + list(cat_cols) + ['target']
    df = df[ordered_cols].copy()
    bin_edges = _fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
    df_mst = _apply_numeric_binning(df, num_cols, bin_edges)
    for c in cat_cols:
        df_mst[c] = df_mst[c].astype(str)
    df_mst['target'] = df_mst['target'].astype(str)
    synth = Synthesizer.create('mst', epsilon=float(eps_cluster), verbose=False)
    synth.fit(df_mst, preprocessor_eps=0.0)
    out = synth.sample(int(n_rows)).copy()
    out = _decode_binned_numeric(out, num_cols, bin_edges)
    for c in cat_cols:
        out[c] = out[c].astype(df[c].dtype if c in df.columns else str)
    out['target'] = pd.to_numeric(out['target'], errors='coerce').fillna(0).astype(int)
    return out[ordered_cols].reset_index(drop=True)

def mst_sample_features_only(df_cluster, num_cols, cat_cols, eps_cluster, n_rows, fixed_label, cat_domains, n_bins=8):
    df = df_cluster.copy()
    ordered_feature_cols = list(num_cols) + list(cat_cols)
    ordered_cols = ordered_feature_cols + ['target']
    if len(df) == 0:
        return pd.DataFrame(columns=ordered_cols)
    df = clip_numeric_df(df, num_cols, 0.0, 1.0)
    df = df[ordered_feature_cols].copy()
    bin_edges = _fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
    df_mst = _apply_numeric_binning(df, num_cols, bin_edges)
    for c in cat_cols:
        if c not in cat_domains:
            raise ValueError(f'Missing public categorical domain for column: {c}')
        df_mst[c] = pd.Categorical(df_mst[c].astype(str), categories=list(map(str, cat_domains[c])))
    synth = Synthesizer.create('mst', epsilon=float(eps_cluster), verbose=False)
    synth.fit(df_mst, preprocessor_eps=0.0)
    out = synth.sample(int(n_rows)).copy()
    out = _decode_binned_numeric(out, num_cols, bin_edges)
    for c in cat_cols:
        out[c] = out[c].astype(df[c].dtype if c in df.columns else str)
    out['target'] = int(fixed_label)
    return out[ordered_cols].reset_index(drop=True)

def strat_C1_global_synth(cfg, row_dict, synth_df, num_cols, cat_cols, k_retr, seed):
    k = min(k_retr, len(synth_df))
    shots = sample_balanced_shots(synth_df, k=k, seed=seed) if k > 0 else None
    if shots is not None:
        print('[DEBUG C1] shot label counts:', shots['target'].value_counts().to_dict())
    return execute_classification_strategy(cfg, row_dict, shots, num_cols + cat_cols)

def strat_C2a_clustered_synth(cfg, row_dict, dp_centroids, synth_store, num_cols, cat_cols, k_retr, seed):
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
        print('[DEBUG C2a] seed:', seed, '| best cluster:', best_cluster, '| shot label counts:', shots['target'].value_counts().to_dict(), '| first rows:', shots.head(2).to_dict('records'))
    if shots is None or len(shots) == 0:
        return 0
    return execute_classification_strategy(cfg, row_dict, shots, num_cols + cat_cols)

def run_epsilon_protocol(datasets_to_run=None, n_trials=20, k_list=None, eps_list=None, results_csv='epsilon_protocol_results.csv'):
    datasets = datasets_to_run or DATASETS_TO_RUN
    k_values = k_list or K_LIST
    eps_values = eps_list or [0.1, 0.5, 1.0, 2.0, 5.0]
    system_msg = 'Follow the Input -> Label pattern. Output only 1 or 0. No explanation.'
    
    for dataset_name in datasets:
        check_data_exists(dataset_name)
        cfg = DATASET_META[dataset_name]
        print(f'\n=== DATASET: {dataset_name} ===')
        train_raw = pd.read_csv(cfg['path'])
        test_raw = pd.read_csv(cfg['test'])
        train_df, num_cols, cat_cols = prep_dataset(train_raw, cfg, dataset_name)
        test_df, _, _ = prep_dataset(test_raw, cfg, dataset_name)
        N_SYN_GLOBAL = len(train_df)
        print("[SYNTH SIZE]", dataset_name, "| tra.in rows =", len(train_df), "| N_SYN_GLOBAL =", N_SYN_GLOBAL)
        
        schema = load_dataset_schema_json(dataset_name)
        cat_domains, num_bounds = schema_to_domains_and_bounds(schema)
        cfg = dict(cfg)
        cfg['name'] = dataset_name
        cfg['schema'] = schema
        cfg['cat_domains'] = cat_domains
        cfg['num_bounds'] = num_bounds
        feat_cols = [c for c in train_df.columns if c != 'target']
        num_cols = [c for c in feat_cols if c in num_bounds]
        cat_cols = [c for c in feat_cols if c in cat_domains and c not in num_cols]
        cfg['name'] = dataset_name
        taskaware_weights = build_taskaware_feature_weights(cfg, num_cols, cat_cols, schema=schema)
        
        for c in cat_cols:
            train_df[c] = train_df[c].astype(str)
            test_df[c] = test_df[c].astype(str)
            
        if len(num_cols) > 0:
            train_df = clip_and_scale_numeric(train_df, num_cols, num_bounds)
            test_df = clip_and_scale_numeric(test_df, num_cols, num_bounds)
            
        if 'C3_PRIVSYN_ICL_DP_SAMPLE_LLM' in METHODS:
            c3_weight_df_fixed, c3_main_df_fixed = split_disjoint_weight_sample(train_df=train_df, sample_frac=WEIGHT_SAMPLE_FRAC, seed=WEIGHT_SPLIT_SEED)
            print("[FIXED DISJOINT SPLIT]", "dataset=", dataset_name, "| split_seed=", WEIGHT_SPLIT_SEED, "| weight_rows=", len(c3_weight_df_fixed), "| main_rows=", len(c3_main_df_fixed))
        else:
            c3_weight_df_fixed = None
            c3_main_df_fixed = None
            
        eval_df = make_balanced_eval(test_df, per_class=EVAL_PER_CLASS, seed=777)
        if len(eval_df) == 0:
            continue

        y_true = eval_df['target'].astype(int).tolist()
        print("[EVAL QUERY LABELS]", eval_df["target"].value_counts().to_dict())
        
        clean_pool = build_clean_pool(train_df, k_pool=min(5000, len(train_df)), seed=42)
        
        for K in k_values:
            X_num = train_df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
            X_cat = np.stack([train_df[c].astype('category').cat.codes.values for c in cat_cols], axis=1).astype(float) if len(cat_cols) > 0 else np.empty((len(train_df), 0), dtype=float)
            X_all = np.concatenate([X_num, X_cat], axis=1)
            clusters = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=CLUSTER_SEED).fit(X_all).labels_
            
            for eps in eps_values:
                print(f'\n>>> K={K} | Epsilon={eps} <<<')
                eps_c2a_cent = 0.1 * eps
                eps_c2a_mem = 0.1 * eps
                eps_c2a_syn = 0.8 * eps
                eps_dpkm = 0.5 * eps
                eps_dpsyn = 0.5 * eps
                
                c3_dp_sample_weights_cached = None
                c3_main_df_cached = c3_main_df_fixed
                c3_weight_df_len = len(c3_weight_df_fixed) if c3_weight_df_fixed is not None else 0
                
                if 'C3_PRIVSYN_ICL_DP_SAMPLE_LLM' in METHODS:
                    qwen_cache_dir = 'qwen_dp_weights_cache'
                    os.makedirs(qwen_cache_dir, exist_ok=True)
                    cache_filename = os.path.join(qwen_cache_dir, f"{dataset_name.lower()}_eps_{float(eps):g}.json")
                    summary_seed = WEIGHT_SUMMARY_BASE_SEED + int(round(float(eps) * 1000))
                    expected_features = set(num_cols + cat_cols)
                
                    if os.path.exists(cache_filename):
                        print(f"--- Loading Cached Qwen Weights for dataset={dataset_name}, EPS={eps} from {cache_filename} ---")
                        with open(cache_filename, 'r') as f:
                            cache_payload = json.load(f)
                        c3_dp_sample_weights_cached = cache_payload["normalized_weights"]
                        if float(cache_payload["epsilon"]) != float(eps): raise ValueError(f"Epsilon mismatch in {cache_filename}")
                        if int(cache_payload["weight_split_seed"]) != WEIGHT_SPLIT_SEED: raise ValueError(f"Split-seed mismatch in {cache_filename}")
                        if set(c3_dp_sample_weights_cached) != expected_features: raise ValueError(f"Feature mismatch in {cache_filename}")
                    else:
                        print(f"--- Generating NEW Qwen Weights for dataset={dataset_name}, EPS={eps} ---")
                        rng_weight_eps = np.random.RandomState(summary_seed)
                        dp_weight_demos_eps = build_dp_weighting_demos(weight_df=c3_weight_df_fixed, num_cols=num_cols, cat_cols=cat_cols, epsilon=eps, rng=rng_weight_eps, cat_domains=cat_domains, dataset_name=dataset_name, num_bounds=num_bounds)
                        c3_dp_sample_weights_cached = build_llm_weights_from_dp_demos(cfg=cfg, dp_demos=dp_weight_demos_eps, num_cols=num_cols, cat_cols=cat_cols, schema=schema)
                        if set(c3_dp_sample_weights_cached) != expected_features: raise ValueError("Qwen returned an invalid feature set.")
                        
                        cache_payload = {
                            "dataset": dataset_name, "epsilon": float(eps), "weight_sample_fraction": float(WEIGHT_SAMPLE_FRAC),
                            "weight_split_seed": int(WEIGHT_SPLIT_SEED), "summary_seed": int(summary_seed),
                            "scoring_model": QWEN_MODEL_NAME, "prompt_version": "qwen_dp_summary_weights_v1",
                            "dp_summaries": dp_weight_demos_eps.to_dict(orient="records"),
                            "normalized_weights": {f: float(w) for f, w in c3_dp_sample_weights_cached.items()}
                        }
                        with open(cache_filename, 'w') as f:
                            json.dump(cache_payload, f, indent=2)
                        print(f"[SAVED] Qwen weights written to {cache_filename}")
                
                    print(f"\n[QWEN DP-GUIDED WEIGHTS] dataset={dataset_name} epsilon={eps} split_seed={WEIGHT_SPLIT_SEED} summary_seed={summary_seed}")

                for trial_seed in [1000 + i for i in range(n_trials)]:
                    rng_b1 = np.random.RandomState(trial_seed + 101)
                    rng_c1 = np.random.RandomState(trial_seed + 202)
                    rng_c2 = np.random.RandomState(trial_seed + 303)
                    
                    b1_pool = None
                    gdp_demos = None
                    c1_synth = None
                    c2_cent, c2_store = (None, None)
                    c2b_cent, c2b_store = (None, None)
                    c2adp_centers, c2adp_store = (None, None)
                    c2adp_ta_construct_centers, c2adp_ta_construct_store = (None, None)
                    c2_global_aim_synth = None
                    c3_global_privsyn_synth = None
                    c3_global_privsyn_target_only_synth = None
                    c3_privsyn_ta_construct_centers, c3_privsyn_ta_construct_store = (None, None)
                    c3_privsyn_icl_construct_centers, c3_privsyn_icl_construct_store, c3_privsyn_icl_selection_log = (None, None, None)
                    c3_dp_sample_centers, c3_dp_sample_store, c3_dp_sample_log, c3_dp_sample_weights, c3_dp_sample_onehot_spec = (None, None, None, None, None)
                    if UNWEIGHTED_ABLATION:
                      c3_dp_sample_weights_cached = {
                          c: 1.0 for c in num_cols + cat_cols
                      }

                      print(
                          "[UNWEIGHTED ABLATION]",
                          sorted(set(c3_dp_sample_weights_cached.values()))
                      )
                    # --- BUILD POOLS ---
                    if 'B1' in METHODS:
                        set_seed(trial_seed + 101)
                        b1_pool = build_B1_ldp_pool(train_df, num_cols, cat_cols, eps, rng_b1, cat_domains)

                    if any((m in METHODS for m in ['C3_GLOBAL_PRIVSYN', 'C3_GLOBAL_PRIVSYN_TA_RETRIEVE'])):
                        set_seed(trial_seed + 858)
                        c3_global_privsyn_synth = build_C3_global_privsyn_pool(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, eps_total=eps, n_synth=N_SYN_GLOBAL, n_bins=8)

                    if 'C3_PRIVSYN_TA_CONSTRUCT' in METHODS:
                        set_seed(trial_seed + 979)
                        rng_c3_ta_construct = np.random.RandomState(trial_seed + 969)
                        c3_privsyn_ta_construct_centers, c3_privsyn_ta_construct_store = build_C3_privsyn_ta_construct(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, eps_cluster=eps_dpkm, eps_synth=eps_dpsyn, k=K, rng=rng_c3_ta_construct, rows_per_cluster=max(1, N_SYN_GLOBAL // max(K, 1)), feature_weights=taskaware_weights, random_state=trial_seed + 969, n_bins=8)

                    if 'C3_PRIVSYN_ICL_DP_SAMPLE_LLM' in METHODS:
                        method_seed = trial_seed + METHOD_SEED_OFFSETS['C3_PRIVSYN_ICL_DP_SAMPLE_LLM']
                        set_seed(method_seed)
                        rng_main = np.random.RandomState(method_seed + 1)
                        eps_main_cluster = 0.40 * eps
                        eps_main_synth = 0.60 * eps
                        c3_dp_sample_centers, c3_dp_sample_store, c3_dp_sample_log, c3_dp_sample_onehot_spec = build_C3_privsyn_icl_all_star(
                            train_df=c3_main_df_cached, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains, 
                            eps_cluster=eps_main_cluster, eps_synth=eps_main_synth, k=K, rng=rng_main,
                            rows_per_cluster=max(1, N_SYN_GLOBAL // max(K, 1)), feature_weights=c3_dp_sample_weights_cached,
                            random_state=method_seed + 1, n_bins=8
                        )
                    if 'C3_GLOBAL_PRIVSYN_TARGET_ONLY' in METHODS:
                      method_seed = (
                          trial_seed
                          + METHOD_SEED_OFFSETS['C3_GLOBAL_PRIVSYN_TARGET_ONLY']
                      )
                      set_seed(method_seed)

                      c3_global_privsyn_target_only_synth = (
                          build_C3_global_privsyn_pool(
                              train_df=train_df,
                              num_cols=num_cols,
                              cat_cols=cat_cols,
                              cat_domains=cat_domains,
                              eps_total=eps,
                              n_synth=N_SYN_GLOBAL,
                              n_bins=8,
                              label_target_only=True,
                          )
                      )

                    if 'C3_PRIVSYN_ICL_CONSTRUCT' in METHODS:
                        set_seed(trial_seed + 979)
                        rng_c3_icl_construct = np.random.RandomState(trial_seed + 979)
                        eps_c3_icl_cluster = 0.2 * eps
                        eps_c3_icl_synth = 0.8 * eps
                        c3_privsyn_icl_construct_centers, c3_privsyn_icl_construct_store, c3_privsyn_icl_selection_log = build_C3_privsyn_icl_construct(
                            train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, cat_domains=cat_domains,
                            eps_cluster=eps_c3_icl_cluster, eps_synth=eps_c3_icl_synth, k=K, rng=rng_c3_icl_construct,
                            rows_per_cluster=max(1, N_SYN_GLOBAL // max(K, 1)), feature_weights=taskaware_weights, random_state=trial_seed + 979, n_bins=8
                        )

                    if 'C1' in METHODS:
                        set_seed(trial_seed + 202)
                        c1_synth = build_C1_mst_pool(train_df, num_cols, cat_cols, eps, n_synth=N_SYN_GLOBAL)

                    if any((m in METHODS for m in ['C2aDP', 'C2aDP_TA_RETRIEVE'])):
                        rng_c2adp = np.random.RandomState(trial_seed + 505)
                        c2adp_centers, c2adp_store = build_C2aDP_diffprivlib_kmeans_mst(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, eps_cluster=eps_dpkm, eps_synth=eps_dpsyn, k=K, rng=rng_c2adp, rows_per_cluster=max(1, N_SYN_GLOBAL // max(K, 1)), cat_domains=cat_domains, random_state=trial_seed + 505, min_cluster_size=MIN_CLUSTER_SIZE)

                    if any((m in METHODS for m in ['C2aDP_TA_CLUSTER', 'C2aDP_TA_BOTH'])):
                        rng_c2adp_ta = np.random.RandomState(trial_seed + 707)
                        c2adp_ta_centers, c2adp_ta_store = build_C2aDP_taskaware_kmeans_mst(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, eps_cluster=eps_dpkm, eps_synth=eps_dpsyn, k=K, rng=rng_c2adp_ta, rows_per_cluster=max(1, N_SYN_GLOBAL // max(K, 1)), cat_domains=cat_domains, feature_weights=taskaware_weights, random_state=trial_seed + 707, min_cluster_size=MIN_CLUSTER_SIZE)

                    if 'C2aDP_TA_CONSTRUCT' in METHODS:
                        rng_c2adp_ta_construct = np.random.RandomState(trial_seed + 909)
                        c2adp_ta_construct_centers, c2adp_ta_construct_store = build_C2aDP_ta_construct(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, eps_cluster=eps_dpkm, eps_synth=eps_dpsyn, k=K, rng=rng_c2adp_ta_construct, rows_per_cluster=max(1, N_SYN_GLOBAL // max(K, 1)), cat_domains=cat_domains, feature_weights=taskaware_weights, random_state=trial_seed + 909, min_cluster_size=MIN_CLUSTER_SIZE, ta_mst_top_m=None)

                    if 'GDP' in METHODS:
                        set_seed(trial_seed + 151)
                        rng_gdp = np.random.RandomState(trial_seed + 151)
                        gdp_demos = build_GDP_tabicl_demos(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, eps_total=eps, k_shots=SHOT_COUNTS[0], rng=rng_gdp, cat_domains=cat_domains, dataset_name=dataset_name, num_bounds=num_bounds, sample_rate=1.0)

                    if 'C2_GLOBAL_AIM' in METHODS:
                        set_seed(trial_seed + 808)
                        c2_global_aim_synth = build_C2_global_aim_pool(train_df=train_df, num_cols=num_cols, cat_cols=cat_cols, eps_total=eps, n_synth=N_SYN_GLOBAL, n_bins=8)

                    # --- EVALUATION ---
                    for method in METHODS:
                        preds = []
                        shots_used = SHOT_COUNTS[0]
                        retrieval_rows = []
                        
                        for ridx, (_, row) in enumerate(eval_df.iterrows()):
                            row_dict = {col: row[col] for col in num_cols + cat_cols}
                            row_seed = trial_seed * 10000 + ridx
                            selected_cluster_for_log = None
                            
                            if method == 'A2':
                                shots = sample_balanced_shots(clean_pool, k=shots_used, seed=row_seed)
                            elif method == 'B1':
                                shots = sample_balanced_shots(b1_pool, k=shots_used, seed=row_seed)
                            elif method == 'C2_GLOBAL_AIM':
                                shots = sample_balanced_shots(c2_global_aim_synth, k=shots_used, seed=row_seed, label_col='target')
                            elif method == 'C3_GLOBAL_PRIVSYN':
                                shots = sample_balanced_shots(c3_global_privsyn_synth, k=shots_used, seed=row_seed, label_col='target')
                            elif method == 'C3_GLOBAL_PRIVSYN_TARGET_ONLY':
                              shots = sample_balanced_shots(
                                  c3_global_privsyn_target_only_synth,
                                  k=shots_used,
                                  seed=row_seed,
                                  label_col='target',
                              )
                            elif method == 'C3_GLOBAL_PRIVSYN_TA_RETRIEVE':
                                shots = retrieve_balanced_nearest_shots_taskaware(c3_global_privsyn_synth, row_dict, k=shots_used, num_cols=num_cols, cat_cols=cat_cols, feature_weights=taskaware_weights, seed=row_seed, label_col='target')
                            elif method == 'C3_PRIVSYN_TA_CONSTRUCT':
                                qvec = encoded_query_vector_taskaware(row_dict, num_cols, cat_cols, cat_domains, taskaware_weights)
                                valid_clusters = [cid for cid, pool in c3_privsyn_ta_construct_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c3_privsyn_ta_construct_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = sample_balanced_shots(c3_privsyn_ta_construct_store[best_c], k=shots_used, seed=row_seed, label_col='target')
                            elif method == "C3_PRIVSYN_ICL_DP_SAMPLE_LLM":
                                qvec = encode_weighted_onehot_query(row_dict, c3_dp_sample_onehot_spec)
                                valid_clusters = [cid for cid, pool in c3_dp_sample_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c3_dp_sample_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = sample_balanced_shots(c3_dp_sample_store[best_c], k=shots_used, seed=row_seed, label_col="target")
                            elif method == 'C3_PRIVSYN_ICL_CONSTRUCT':
                                qvec = encoded_query_vector_taskaware(row_dict, num_cols, cat_cols, cat_domains, taskaware_weights)
                                valid_clusters = [cid for cid, pool in c3_privsyn_icl_construct_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c3_privsyn_icl_construct_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = sample_balanced_shots(c3_privsyn_icl_construct_store[best_c], k=shots_used, seed=row_seed, label_col='target')
                            elif method == 'C1':
                                shots = sample_balanced_shots(c1_synth, k=shots_used, seed=row_seed)
                            elif method == 'C2aDP':
                                qvec = encoded_query_vector(row_dict, num_cols, cat_cols, cat_domains)
                                valid_clusters = [cid for cid, pool in c2adp_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c2adp_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = sample_balanced_shots(c2adp_store[best_c], k=shots_used, seed=row_seed)
                            elif method == 'C2aDP_TA_CLUSTER':
                                qvec = encoded_query_vector_taskaware(row_dict, num_cols, cat_cols, cat_domains, taskaware_weights)
                                valid_clusters = [cid for cid, pool in c2adp_ta_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c2adp_ta_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = sample_balanced_shots(c2adp_ta_store[best_c], k=shots_used, seed=row_seed)
                            elif method == 'C2aDP_TA_RETRIEVE':
                                qvec = encoded_query_vector(row_dict, num_cols, cat_cols, cat_domains)
                                valid_clusters = [cid for cid, pool in c2adp_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c2adp_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = retrieve_balanced_nearest_shots_taskaware(c2adp_store[best_c], row_dict, k=shots_used, num_cols=num_cols, cat_cols=cat_cols, feature_weights=taskaware_weights, seed=row_seed)
                            elif method == 'C2aDP_TA_BOTH':
                                qvec = encoded_query_vector_taskaware(row_dict, num_cols, cat_cols, cat_domains, taskaware_weights)
                                valid_clusters = [cid for cid, pool in c2adp_ta_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c2adp_ta_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = retrieve_balanced_nearest_shots_taskaware(c2adp_ta_store[best_c], row_dict, k=shots_used, num_cols=num_cols, cat_cols=cat_cols, feature_weights=taskaware_weights, seed=row_seed)
                            elif method == 'C2aDP_TA_CONSTRUCT':
                                qvec = encoded_query_vector_taskaware(row_dict, num_cols, cat_cols, cat_domains, taskaware_weights)
                                valid_clusters = [cid for cid, pool in c2adp_ta_construct_store.items() if pool is not None and len(pool) > 0]
                                if len(valid_clusters) == 0:
                                    shots = None
                                else:
                                    best_c = min(valid_clusters, key=lambda cid: float(((qvec - c2adp_ta_construct_centers[cid]) ** 2).sum()))
                                    selected_cluster_for_log = int(best_c)
                                    shots = sample_balanced_shots(c2adp_ta_construct_store[best_c], k=shots_used, seed=row_seed)
                            elif method == 'GDP':
                                pred = strat_GDP_tabicl(cfg, row_dict, gdp_demos, num_cols, cat_cols)
                                preds.append(pred)
                                continue
                            else:
                                raise ValueError(f'Unknown method: {method}')
                                
                            if shots is None or len(shots) == 0:
                                pred = 0
                            else:
                                pred = execute_classification_strategy(cfg, row_dict, shots, num_cols + cat_cols)
                                
                            preds.append(int(pred))
                            if SAVE_SYNTH_ANALYSIS:
                                log_row = {'dataset': dataset_name, 'method': method, 'epsilon': eps, 'K': K, 'seed': trial_seed, 'query_id': ridx, 'true_label': int(row['target']), 'pred_label': int(pred), 'correct': int(int(pred) == int(row['target'])), 'selected_cluster': selected_cluster_for_log, 'shots_used': len(shots) if shots is not None else 0}
                                if shots is not None and len(shots) > 0 and ('target' in shots.columns):
                                    log_row['shot_label_counts'] = json.dumps({str(k): int(v) for k, v in shots['target'].value_counts().to_dict().items()})
                                    log_row['shot_labels'] = json.dumps([int(x) for x in shots['target'].tolist()])
                                retrieval_rows.append(log_row)
                                
                        f1 = f1_score(y_true, preds)
                        acc = accuracy_score(y_true, preds)
                        prec = precision_score(y_true, preds, zero_division=0)
                        rec = recall_score(y_true, preds, zero_division=0)
                        row_out = {'dataset': dataset_name, 'K': K, 'epsilon': eps, 'shots': shots_used, 'seed': trial_seed, 'method': method, 'top_r_clusters': TOP_R_CLUSTERS, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
                        
                        if method == 'C3_PRIVSYN_ICL_DP_SAMPLE_LLM':
                            row_out['weight_sample_frac'] = WEIGHT_SAMPLE_FRAC
                            row_out['weight_source'] = 'disjoint_dp_examples_plus_llm'
                            row_out['avg_features_kept'] = len(num_cols) + len(cat_cols)
                            row_out['eps_cluster_used'] = round(eps_main_cluster, 4)
                            row_out['eps_synth_used'] = round(eps_main_synth, 4)
                        elif method == 'C3_PRIVSYN_ICL_CONSTRUCT' and c3_privsyn_icl_selection_log:
                            kept_counts = [len(log.get('used_features', [])) for log in c3_privsyn_icl_selection_log.values()]
                            avg_kept = sum(kept_counts) / max(len(kept_counts), 1)
                            row_out['avg_features_kept'] = round(avg_kept, 1)
                        else:
                            row_out['avg_features_kept'] = len(num_cols) + len(cat_cols)
                            
                        append_row_to_csv(row_out, results_csv)
                        print(f'[SAVED] {dataset_name} | eps={eps} | K={K} | seed={trial_seed} | method={method} | f1={f1:.3f}')
                        
                        if SAVE_SYNTH_ANALYSIS:
                            save_retrieval_log_csv(out_root=SYNTH_ANALYSIS_DIR, dataset_name=dataset_name, method=method, epsilon=eps, K=K, seed=trial_seed, retrieval_rows=retrieval_rows)
                            
                    # --- MEMORY CLEANUP ---
                    if 'b1_pool' in locals(): del b1_pool
                    if 'c1_synth' in locals(): del c1_synth
                    if 'c2_cent' in locals(): del c2_cent
                    if 'c2_store' in locals(): del c2_store
                    if 'c2b_cent' in locals(): del c2b_cent
                    if 'c2b_store' in locals(): del c2b_store
                    if 'c2adp_centers' in locals(): del c2adp_centers
                    if 'c2adp_store' in locals(): del c2adp_store
                    if 'c2adp_ta_centers' in locals(): del c2adp_ta_centers
                    if 'c2adp_ta_store' in locals(): del c2adp_ta_store
                    if 'c2adp_ta_construct_centers' in locals(): del c2adp_ta_construct_centers
                    if 'c2adp_ta_construct_store' in locals(): del c2adp_ta_construct_store
                    if 'c3_global_privsyn_synth' in locals(): del c3_global_privsyn_synth
                    cleanup()

def main():
    global METHODS, WEIGHT_SAMPLE_FRAC
    parser = argparse.ArgumentParser(description='DP-GAN server runner')
    parser.add_argument('--mode', choices=['epsilon', 'run_all'], default='epsilon')
    parser.add_argument('--datasets', nargs='*', default=None)
    parser.add_argument('--trials', type=int, default=N_TRIALS)
    parser.add_argument('--k_list', nargs='*', type=int, default=K_LIST)
    parser.add_argument('--eps_list', nargs='*', type=float, default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument('--results_csv', default='epsilon_protocol_results.csv')
    parser.add_argument('--methods', nargs='*', default=None)
    parser.add_argument('--weight_sample_frac', type=float, default=WEIGHT_SAMPLE_FRAC)
    args = parser.parse_args()
    if args.methods:
        METHODS = list(args.methods)
    WEIGHT_SAMPLE_FRAC = float(args.weight_sample_frac)
    df = run_epsilon_protocol(datasets_to_run=args.datasets, n_trials=args.trials, k_list=args.k_list, eps_list=args.eps_list, results_csv=args.results_csv)
if __name__ == '__main__':
    main()
