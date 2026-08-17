#!/usr/bin/env python3
"""
Methods
-------
GLOBAL_GEM
    Official GEM on the complete private dataset with the full privacy budget.
    No clustering and no DP-summary/Qwen weighting branch.

C3_GEM_ICL_DP_SAMPLE_LLM
    Exact C3 data path from the current base experiment:
      fixed disjoint 10% weighting partition / 90% main partition ->
      DP class-conditioned summaries -> Qwen data-guided feature weights ->
      base weighted one-hot diffprivlib DP-KMeans on the main partition ->
      target-centered official GEM independently on each disjoint non-empty cluster ->
      the SAME DP-centroid prototype fallback used by the current base for empty/failed clusters ->
      nearest-DP-centroid routing -> balanced demonstration sampling -> Llama inference.

Official GEM is loaded directly from terranceliu/iterative-dp.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import itertools
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_BASE_STEMS = (
    "Pritabicl_Privsyn_Main.py",
)

METHOD_GLOBAL = "GLOBAL_GEM"
METHOD_CLUSTER_TARGET = "CLUSTER_TARGET_GEM"
METHOD_C3_SAMPLE_DP_LLM = "C3_GEM_ICL_DP_SAMPLE_LLM"
ALL_METHODS = (METHOD_GLOBAL, METHOD_C3_SAMPLE_DP_LLM)


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_base_script(explicit_path: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("PRITABICL_BASE_SCRIPT")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    roots = [Path.cwd(), Path(__file__).resolve().parent, Path("/mnt/data")]
    for root in roots:
        for name in DEFAULT_BASE_STEMS:
            candidates.append(root / name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find the unchanged PriTabICL base script. "
        "Pass --base-script or set PRITABICL_BASE_SCRIPT.\n"
        f"Searched:\n{searched}"
    )


def load_base_module(base_script: Path):
    return _load_module(base_script, "pritabicl_base_gem_variants")


def find_gem_home(explicit_path: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.environ.get("GEM_HOME")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([Path("/content/iterative-dp"), Path.cwd() / "iterative-dp"])
    for candidate in candidates:
        if (candidate / "gem.py").is_file() and (candidate / "Util" / "qm.py").is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find the official iterative-dp GEM repository. "
        "Pass --gem-home or set GEM_HOME.\n"
        f"Searched:\n{searched}"
    )


def load_official_gem(gem_home: Path):
    """Import GEM and helpers directly from the official repository."""
    gem_home = gem_home.resolve()
    if str(gem_home) not in sys.path:
        sys.path.insert(0, str(gem_home))
    # Compatibility with newer NumPy releases used by Colab.
    if not hasattr(np, "infty"):
        np.infty = np.inf
    try:
        import gem as gem_module
        from Util.qm import QueryManager
        from Util import util_general
        from Util.util_gem import get_synth_data
        from mbi import Dataset, Domain
    except Exception as exc:
        raise ImportError(
            "Unable to import the official GEM implementation. Clone "
            "https://github.com/terranceliu/iterative-dp into /content/iterative-dp "
            "and install its requirements."
        ) from exc
    return gem_module, QueryManager, util_general, get_synth_data, Dataset, Domain


def find_google_dp_learning_home(explicit_path: Optional[str] = None) -> Path:
    """Find Google's official differential-privacy/learning checkout."""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    env_path = os.environ.get("GOOGLE_DP_LEARNING_HOME")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([
        Path("/content/differential-privacy/learning"),
        Path.cwd() / "differential-privacy" / "learning",
    ])
    for candidate in candidates:
        if (candidate / "clustering" / "clustering_algorithm.py").is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find Google's official DP clustering library. Clone "
        "https://github.com/google/differential-privacy.git into "
        "/content/differential-privacy and pass --google-dp-learning-home if needed.\n"
        f"Searched:\n{searched}"
    )


def load_google_dp_clustering(learning_home: Path):
    """Import Google's official practical central-DP clustering code."""
    learning_home = learning_home.resolve()
    if str(learning_home) not in sys.path:
        sys.path.insert(0, str(learning_home))
    try:
        from clustering import clustering_algorithm
        from clustering import clustering_params
    except Exception as exc:
        raise ImportError(
            "Unable to import Google's official clustering package from "
            f"{learning_home}. Install the learning package/dependencies, e.g. "
            "`pip install -e /content/differential-privacy/learning`."
        ) from exc
    return clustering_algorithm, clustering_params


def fit_google_dp_clustering_onehot_weighted(
    base, google_components, df, num_cols, cat_cols, cat_domains,
    feature_weights, k, eps_cluster, delta_cluster, random_state,
):
    clustering_algorithm, clustering_params = google_components
    if float(eps_cluster) <= 0:
        raise ValueError("eps_cluster must be positive")
    if not 0.0 < float(delta_cluster) < 1.0:
        raise ValueError("delta_cluster must lie in (0,1)")

    spec = base.build_weighted_onehot_spec(
        num_cols=list(num_cols),
        cat_cols=list(cat_cols),
        cat_domains=cat_domains,
        feature_weights=feature_weights,
    )
    Xw = base.encode_weighted_onehot_df(df, spec)
    if len(Xw) == 0:
        raise ValueError("Cannot cluster an empty dataframe")

    radius_sq = 0.0
    for col in spec["num_cols"]:
        radius_sq += max(float(spec["feature_weights"].get(col, 1.0)), 0.0)
    for col in spec["cat_cols"]:
        width = max(len(spec["cat_domains"][col]), 1)
        radius_sq += max(float(spec["feature_weights"].get(col, 1.0)), 0.0) / float(width)
    radius = max(float(np.sqrt(radius_sq)), 1e-8)

    np.random.seed(int(random_state))
    data = clustering_params.Data(Xw, radius)
    privacy_param = clustering_params.DifferentialPrivacyParam(
        epsilon=float(eps_cluster), delta=float(delta_cluster)
    )
    result = clustering_algorithm.private_lsh_clustering(
        int(k), data, privacy_param
    )
    centers = np.asarray(result.centers, dtype=float)
    labels = np.asarray(result.labels, dtype=int)
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    print(
        "[GOOGLE DP CLUSTERING]",
        "shape:", Xw.shape,
        "| requested_K:", int(k),
        "| returned_K:", len(centers),
        "| radius:", f"{radius:.6f}",
        "| counts:", counts,
    )
    return centers, labels, spec


def _make_domain(Domain, config: Dict[str, int]):
    if hasattr(Domain, "fromdict"):
        return Domain.fromdict(config)
    return Domain(list(config.keys()), list(config.values()))


def _encode_for_gem(
    train_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_domains: Dict[str, Sequence[str]],
    n_bins: int,
):
    ordered_cols = list(num_cols) + list(cat_cols) + ["target"]
    encoded = pd.DataFrame(index=train_df.index)
    decode_info = {"numeric": {}, "categorical": {}}

    for col in num_cols:
        values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        codes = np.digitize(values.to_numpy(), edges[1:-1], right=False)
        encoded[col] = codes.astype(int)
        decode_info["numeric"][col] = edges

    for col in cat_cols:
        domain = list(map(str, cat_domains[col]))
        mapping = {value: idx for idx, value in enumerate(domain)}
        unknown = sorted(set(train_df[col].astype(str)) - set(domain))
        if unknown:
            raise ValueError(f"{col}: values missing from public schema domain: {unknown[:10]}")
        encoded[col] = train_df[col].astype(str).map(mapping).astype(int)
        decode_info["categorical"][col] = domain

    encoded["target"] = pd.to_numeric(train_df["target"], errors="raise").astype(int)
    encoded = encoded[ordered_cols].reset_index(drop=True)
    domain_config = {col: int(encoded[col].max()) + 1 for col in ordered_cols}
    domain_config["target"] = 2
    return encoded, domain_config, decode_info


def _decode_gem_output(
    synth_codes: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    decode_info: dict,
) -> pd.DataFrame:
    out = pd.DataFrame(index=synth_codes.index)
    for col in num_cols:
        edges = np.asarray(decode_info["numeric"][col], dtype=float)
        mids = (edges[:-1] + edges[1:]) / 2.0
        idx = pd.to_numeric(synth_codes[col], errors="coerce").fillna(0).astype(int)
        idx = idx.clip(0, len(mids) - 1)
        out[col] = mids[idx.to_numpy()]
    for col in cat_cols:
        domain = decode_info["categorical"][col]
        idx = pd.to_numeric(synth_codes[col], errors="coerce").fillna(0).astype(int)
        idx = idx.clip(0, len(domain) - 1)
        out[col] = [domain[i] for i in idx.to_numpy()]
    out["target"] = (
        pd.to_numeric(synth_codes["target"], errors="coerce")
        .fillna(0).round().astype(int).clip(0, 1)
    )
    return out[list(num_cols) + list(cat_cols) + ["target"]].reset_index(drop=True)


def build_gem_pool(
    gem_components,
    train_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_domains: Dict[str, Sequence[str]],
    epsilon: float,
    delta: float,
    n_synth: int,
    n_bins: int,
    target_centered: bool,
    seed: int,
    **gem_kwargs,
) -> pd.DataFrame:
    """Run official GEM with either all 2-way or feature-target workloads."""
    gem_module, QueryManager, util_general, get_synth_data, Dataset, Domain = gem_components
    np.random.seed(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass

    encoded, domain_config, decode_info = _encode_for_gem(
        train_df, num_cols, cat_cols, cat_domains, n_bins
    )
    domain = _make_domain(Domain, domain_config)
    data = Dataset(encoded, domain)
    feature_cols = list(num_cols) + list(cat_cols)
    if target_centered:
        workloads = [(col, "target") for col in feature_cols]
    else:
        degree = int(gem_kwargs.pop("degree", 2))
        workloads = list(itertools.combinations(list(encoded.columns), degree))
    if not workloads:
        raise ValueError("GEM workload is empty.")

    qm = QueryManager(domain, workloads)
    real_answers = qm.get_answer(data, concat=True)
    T = int(gem_kwargs.pop("T", 10))
    alpha = float(gem_kwargs.pop("alpha", 0.5))
    embedding_dim = int(gem_kwargs.pop("embedding_dim", 128))
    batch_size = int(gem_kwargs.pop("batch_size", min(max(int(n_synth), 2), 500)))
    lr = float(gem_kwargs.pop("lr", 1e-4))
    eta_min = gem_kwargs.pop("eta_min", None)
    max_iters = int(gem_kwargs.pop("max_iters", 100))
    max_idxs = int(gem_kwargs.pop("max_idxs", 100))
    resample = bool(gem_kwargs.pop("resample", False))
    verbose = bool(gem_kwargs.pop("verbose", False))
    if gem_kwargs:
        raise ValueError(f"Unknown GEM keyword arguments: {sorted(gem_kwargs)}")

    eps0, rho = util_general.get_eps0_zCDP(
        float(epsilon), float(delta), T, alpha=alpha
    )
    save_dir = Path(tempfile.mkdtemp(prefix="pritabicl_gem_"))
    try:
        model = gem_module.GEM(
            embedding_dim=embedding_dim,
            gen_dim=(embedding_dim * 2, embedding_dim * 2),
            batch_size=batch_size,
            save_dir=str(save_dir),
        )
        model.setup_data(
            encoded,
            discrete_columns=tuple(encoded.columns),
            domain=domain,
            overrides=["transformer"],
        )
        model.fit(
            T=T,
            eps0=eps0,
            sensitivity=1.0 / max(len(encoded), 1),
            qm=qm,
            real_answers=real_answers,
            lr=lr,
            eta_min=eta_min,
            resample=resample,
            max_idxs=max_idxs,
            max_iters=max_iters,
            alpha=alpha,
            save_interval=max(T + 1, 1),
            save_num=1,
            verbose=verbose,
        )
        synth_codes = get_synth_data(model, int(n_synth), dtype=np.int64)
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
    return _decode_gem_output(synth_codes, num_cols, cat_cols, decode_info)


def _validate_pool(
    pool: pd.DataFrame, ordered_cols: Sequence[str], context: str
) -> pd.DataFrame:
    if not isinstance(pool, pd.DataFrame):
        raise TypeError(f"{context}: GEM must return pandas.DataFrame")
    missing = [c for c in ordered_cols if c not in pool.columns]
    if missing:
        raise ValueError(f"{context}: GEM output is missing columns: {missing}")
    out = pool[list(ordered_cols)].copy().reset_index(drop=True)
    out["target"] = pd.to_numeric(out["target"], errors="raise").astype(int)
    if not set(out["target"].unique()).issubset({0, 1}):
        raise ValueError(f"{context}: target must contain only 0 and 1")
    return out


def call_gem(
    gem_components, train_df, num_cols, cat_cols, cat_domains, epsilon, delta,
    n_synth, n_bins, target_centered, seed, gem_kwargs
):
    ordered_cols = list(num_cols) + list(cat_cols) + ["target"]
    pool = build_gem_pool(
        gem_components=gem_components,
        train_df=train_df[ordered_cols].copy().reset_index(drop=True),
        num_cols=list(num_cols), cat_cols=list(cat_cols),
        cat_domains={k: list(v) for k, v in cat_domains.items()},
        epsilon=float(epsilon), delta=float(delta), n_synth=int(n_synth),
        n_bins=int(n_bins), target_centered=bool(target_centered),
        seed=int(seed), **dict(gem_kwargs),
    )
    return _validate_pool(pool, ordered_cols, "GEM")

def _validate_methods(methods: Iterable[str]) -> List[str]:
    methods = list(methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {list(ALL_METHODS)}")
    return methods


def build_cluster_target_gem_pools(
    base,
    gem_components,
    train_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_domains: Dict[str, Sequence[str]],
    feature_weights: Dict[str, float],
    k: int,
    eps_cluster: float,
    eps_synth: float,
    delta_synth: float,
    rows_per_cluster: int,
    random_state: int,
    n_bins: int,
    gem_kwargs: dict,
):
    """Weighted one-hot DP clustering followed by target-centered GEM.

    Privacy accounting:
      * Base weighted one-hot diffprivlib DP-KMeans consumes eps_cluster (pure DP).
      * GEM consumes (eps_synth, delta_synth) on each induced cluster subset.
      * Cluster GEM calls compose in parallel because the subsets are disjoint.
      * Public-schema semantic weights consume no privacy budget.
    """
    if float(eps_cluster) <= 0 or float(eps_synth) <= 0:
        raise ValueError("Both eps_cluster and eps_synth must be positive.")
    if not 0.0 < float(delta_synth) < 1.0:
        raise ValueError("delta_synth must lie in (0, 1).")

    centers_w, labels, onehot_spec = base.fit_diffprivlib_dp_kmeans_onehot_weighted(
        df=train_df,
        num_cols=list(num_cols),
        cat_cols=list(cat_cols),
        cat_domains=cat_domains,
        feature_weights=feature_weights,
        k=int(k),
        eps_cluster=float(eps_cluster),
        random_state=int(random_state),
    )

    ordered_cols = list(num_cols) + list(cat_cols) + ["target"]
    stores: Dict[int, pd.DataFrame] = {}
    logs: Dict[int, dict] = {}
    centers_out: List[np.ndarray] = []
    rows_per_cluster = max(1, int(rows_per_cluster))
    labels_arr = np.asarray(labels)

    for cluster_id in range(len(centers_w)):
        center_w = np.asarray(centers_w[cluster_id], dtype=float)
        cluster_df = (
            train_df[labels_arr == cluster_id].copy().reset_index(drop=True)
        )
        centers_out.append(center_w)
        center_decoded = base.decode_weighted_onehot_center(center_w, onehot_spec)
        fallback = base.make_c2adp_centroid_fallback(
            center=center_decoded,
            num_cols=list(num_cols),
            cat_cols=list(cat_cols),
            cat_domains=cat_domains,
            n_rows=rows_per_cluster,
        )[ordered_cols].reset_index(drop=True)

        if len(cluster_df) == 0:
            stores[cluster_id] = fallback
            logs[cluster_id] = {
                "fallback": True,
                "private_cluster_rows": 0,
                "target_centered": True,
                "encoding": "weighted_onehot_diffprivlib_dp_kmeans",
                "error": "empty_cluster",
            }
            continue

        try:
            seed = int(random_state) + 1000 + cluster_id
            base.set_seed(seed)
            pool = call_gem(
                gem_components=gem_components,
                train_df=cluster_df,
                num_cols=num_cols,
                cat_cols=cat_cols,
                cat_domains=cat_domains,
                epsilon=float(eps_synth),
                delta=float(delta_synth),
                n_synth=rows_per_cluster,
                n_bins=int(n_bins),
                target_centered=True,
                seed=seed,
                gem_kwargs=gem_kwargs,
            )
            stores[cluster_id] = pool
            logs[cluster_id] = {
                "fallback": False,
                "private_cluster_rows": int(len(cluster_df)),
                "target_centered": True,
                "encoding": "weighted_onehot_diffprivlib_dp_kmeans",
            }
        except Exception as exc:
            print(f"[GEM FALLBACK] cluster={cluster_id} error={exc!r}")
            stores[cluster_id] = fallback
            logs[cluster_id] = {
                "fallback": True,
                "private_cluster_rows": int(len(cluster_df)),
                "target_centered": True,
                "encoding": "weighted_onehot_diffprivlib_dp_kmeans",
                "error": repr(exc),
            }

    return np.asarray(centers_out), stores, logs, onehot_spec


def route_to_cluster_pool(
    base,
    row_dict,
    centers_w,
    stores,
    logs,
    onehot_spec,
):
    qvec = base.encode_weighted_onehot_query(
        row_dict,
        onehot_spec,
    )

    # Only clusters with a real synthesizer-generated pool.
    valid = [
        cid
        for cid, pool in stores.items()
        if (
            pool is not None
            and len(pool) > 0
            and not logs.get(cid, {}).get("fallback", False)
        )
    ]

    if not valid:
        return None, None

    best_cluster = min(
        valid,
        key=lambda cid: float(
            np.sum((qvec - centers_w[cid]) ** 2)
        ),
    )

    return int(best_cluster), stores[best_cluster]

def load_or_build_c3_sample_dp_llm_weights(
    base,
    cfg,
    dataset_name: str,
    epsilon: float,
    weight_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_domains: Dict[str, Sequence[str]],
    num_bounds: Dict[str, tuple],
    schema,
    cache_dir: str,
    split_fraction: float,
    split_seed: int,
    summary_base_seed: int,
):
    """Exact C3 sampled-DP-LLM DP-summary -> Qwen weight path."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{dataset_name.lower()}_eps_{float(epsilon):g}.json"
    summary_seed = int(summary_base_seed) + int(round(float(epsilon) * 1000))
    expected_features = set(list(num_cols) + list(cat_cols))

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        weights = {str(k): float(v) for k, v in payload["normalized_weights"].items()}
        if float(payload["epsilon"]) != float(epsilon):
            raise ValueError(f"Epsilon mismatch in {cache_path}")
        if int(payload["weight_split_seed"]) != int(split_seed):
            raise ValueError(f"Split-seed mismatch in {cache_path}")
        if set(weights) != expected_features:
            raise ValueError(f"Feature mismatch in {cache_path}")
        print(f"[C3 WEIGHTS CACHE] loaded {cache_path}")
        return weights, str(cache_path)

    rng = np.random.RandomState(summary_seed)
    dp_demos = base.build_dp_weighting_demos(
        weight_df=weight_df,
        num_cols=list(num_cols),
        cat_cols=list(cat_cols),
        epsilon=float(epsilon),
        rng=rng,
        cat_domains=cat_domains,
        dataset_name=dataset_name,
        num_bounds=num_bounds,
    )
    weights = base.build_llm_weights_from_dp_demos(
        cfg=cfg,
        dp_demos=dp_demos,
        num_cols=list(num_cols),
        cat_cols=list(cat_cols),
        schema=schema,
    )
    if set(weights) != expected_features:
        raise ValueError(
            "Qwen returned an invalid feature set. "
            f"Expected={sorted(expected_features)}, received={sorted(weights)}"
        )

    payload = {
        "dataset": dataset_name,
        "epsilon": float(epsilon),
        "weight_sample_fraction": float(split_fraction),
        "weight_split_seed": int(split_seed),
        "summary_seed": int(summary_seed),
        "scoring_model": getattr(base, "QWEN_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct"),
        "prompt_version": "qwen_dp_summary_weights_v1",
        "dp_summaries": dp_demos.to_dict(orient="records"),
        "normalized_weights": {f: float(w) for f, w in weights.items()},
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[C3 WEIGHTS CACHE] saved {cache_path}")
    return weights, str(cache_path)


def route_to_cluster_pool_dp_centroid_fallback(
    base,
    row_dict,
    centers_w,
    stores,
    logs,
    onehot_spec,
):
    qvec = base.encode_weighted_onehot_query(
        row_dict,
        onehot_spec,
    )

    valid = [
        cid
        for cid, pool in stores.items()
        if pool is not None and len(pool) > 0
    ]

    if not valid:
        return None, None

    best_cluster = min(
        valid,
        key=lambda cid: float(
            np.sum((qvec - centers_w[cid]) ** 2)
        ),
    )

    return int(best_cluster), stores[best_cluster]



def run_experiment(
    base,
    gem_components,
    datasets: Sequence[str],
    methods: Sequence[str],
    trials: int,
    k_list: Sequence[int],
    eps_list: Sequence[float],
    results_csv: str,
    epsilon_cluster_share: float,
    delta: float,
    n_bins: int,
    gem_kwargs: dict,
    weight_sample_frac: float,
    weight_split_seed: int,
    weight_summary_base_seed: int,
    hybrid_weight_cache_dir: str,
):
    methods = _validate_methods(methods)
    if not 0.0 < float(epsilon_cluster_share) < 1.0:
        raise ValueError("epsilon_cluster_share must lie in (0, 1).")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    if not 0.0 < float(weight_sample_frac) < 1.0:
        raise ValueError("weight_sample_frac must lie in (0, 1).")

    for dataset_name in datasets:
        base.check_data_exists(dataset_name)
        cfg0 = base.DATASET_META[dataset_name]
        print(f"\n=== DATASET: {dataset_name} ===")

        train_raw = pd.read_csv(cfg0["path"])
        test_raw = pd.read_csv(cfg0["test"])
        train_df, _, _ = base.prep_dataset(train_raw, cfg0, dataset_name)
        test_df, _, _ = base.prep_dataset(test_raw, cfg0, dataset_name)

        schema = base.load_dataset_schema_json(dataset_name)
        if schema is None:
            raise ValueError(f"No public schema found for {dataset_name}.")
        cat_domains, num_bounds = base.schema_to_domains_and_bounds(schema)

        cfg = dict(cfg0)
        cfg.update(
            name=dataset_name,
            schema=schema,
            cat_domains=cat_domains,
            num_bounds=num_bounds,
        )

        feature_cols = [c for c in train_df.columns if c != "target"]
        num_cols = [c for c in feature_cols if c in num_bounds]
        cat_cols = [
            c for c in feature_cols
            if c in cat_domains and c not in num_cols
        ]
        omitted = sorted(set(feature_cols) - set(num_cols) - set(cat_cols))
        if omitted:
            raise ValueError(
                f"Public schema does not define these features for "
                f"{dataset_name}: {omitted}"
            )

        feature_weights = None
        if METHOD_CLUSTER_TARGET in methods:
            feature_weights = base.build_taskaware_feature_weights(
                cfg,
                num_cols,
                cat_cols,
                schema=schema,
            )

        for col in cat_cols:
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

        if num_cols:
            train_df = base.clip_and_scale_numeric(train_df, num_cols, num_bounds)
            test_df = base.clip_and_scale_numeric(test_df, num_cols, num_bounds)

        c3_weight_df = None
        c3_main_df = None
        if METHOD_C3_SAMPLE_DP_LLM in methods:
            c3_weight_df, c3_main_df = base.split_disjoint_weight_sample(
                train_df=train_df,
                sample_frac=float(weight_sample_frac),
                seed=int(weight_split_seed),
            )
            print(
                "[C3 DISJOINT SPLIT]",
                "dataset=", dataset_name,
                "| split_seed=", weight_split_seed,
                "| weight_rows=", len(c3_weight_df),
                "| main_rows=", len(c3_main_df),
            )

        eval_df = base.make_balanced_eval(
            test_df, per_class=base.EVAL_PER_CLASS, seed=777
        )
        if eval_df is None or len(eval_df) == 0:
            print(f"[SKIP] No balanced evaluation set for {dataset_name}")
            continue

        y_true = eval_df["target"].astype(int).tolist()
        n_synth_global = len(train_df)

        for k in k_list:
            rows_per_cluster = max(1, n_synth_global // max(int(k), 1))

            for epsilon in eps_list:
                eps_cluster = float(epsilon_cluster_share) * float(epsilon)
                eps_synth = float(epsilon) - eps_cluster
                delta_cluster = 0.0
                delta_synth = float(delta)

                c3_weights = None
                c3_weight_cache_path = None
                if METHOD_C3_SAMPLE_DP_LLM in methods:
                    c3_weights, c3_weight_cache_path = load_or_build_c3_sample_dp_llm_weights(
                        base=base,
                        cfg=cfg,
                        dataset_name=dataset_name,
                        epsilon=float(epsilon),
                        weight_df=c3_weight_df,
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                        cat_domains=cat_domains,
                        num_bounds=num_bounds,
                        schema=schema,
                        cache_dir=hybrid_weight_cache_dir,
                        split_fraction=float(weight_sample_frac),
                        split_seed=int(weight_split_seed),
                        summary_base_seed=int(weight_summary_base_seed),
                    )
                    print(
                        "[C3 DP-GUIDED WEIGHTS]", dataset_name,
                        "| epsilon=", epsilon,
                        "| cache=", c3_weight_cache_path,
                        "| weights=", c3_weights,
                    )

                for trial in range(int(trials)):
                    trial_seed = 1000 + trial
                    print(
                        f"\n[{dataset_name}] K={k} eps={epsilon} "
                        f"seed={trial_seed}"
                    )

                    global_pool = None
                    target_centers = target_store = target_log = target_spec = None
                    c3_centers = c3_store = c3_log = c3_spec = None

                    if METHOD_GLOBAL in methods:
                        global_seed = trial_seed + 808
                        base.set_seed(global_seed)
                        global_pool = call_gem(
                            gem_components=gem_components,
                            train_df=train_df,
                            num_cols=num_cols,
                            cat_cols=cat_cols,
                            cat_domains=cat_domains,
                            epsilon=float(epsilon),
                            delta=float(delta),
                            n_synth=n_synth_global,
                            n_bins=int(n_bins),
                            target_centered=False,
                            seed=global_seed,
                            gem_kwargs=gem_kwargs,
                        )

                    if METHOD_CLUSTER_TARGET in methods:
                        (
                            target_centers,
                            target_store,
                            target_log,
                            target_spec,
                        ) = build_cluster_target_gem_pools(
                            base=base,
                            gem_components=gem_components,
                            train_df=train_df,
                            num_cols=num_cols,
                            cat_cols=cat_cols,
                            cat_domains=cat_domains,
                            feature_weights=feature_weights,
                            k=int(k),
                            eps_cluster=eps_cluster,
                            eps_synth=eps_synth,
                            delta_synth=delta_synth,
                            rows_per_cluster=rows_per_cluster,
                            random_state=trial_seed + 1082,
                            n_bins=int(n_bins),
                            gem_kwargs=gem_kwargs,
                        )

                    if METHOD_C3_SAMPLE_DP_LLM in methods:
                        (
                            c3_centers,
                            c3_store,
                            c3_log,
                            c3_spec,
                        ) = build_cluster_target_gem_pools(
                            base=base,
                            gem_components=gem_components,
                            train_df=c3_main_df,
                            num_cols=num_cols,
                            cat_cols=cat_cols,
                            cat_domains=cat_domains,
                            feature_weights=c3_weights,
                            k=int(k),
                            eps_cluster=eps_cluster,
                            eps_synth=eps_synth,
                            delta_synth=delta_synth,
                            rows_per_cluster=rows_per_cluster,
                            random_state=trial_seed + 1082,
                            n_bins=int(n_bins),
                            gem_kwargs=gem_kwargs,
                        )

                    for method in methods:
                        predictions: List[int] = []
                        retrieval_rows: List[dict] = []

                        for row_idx, (_, row) in enumerate(eval_df.iterrows()):
                            row_dict = {
                                col: row[col] for col in num_cols + cat_cols
                            }
                            row_seed = trial_seed * 10000 + row_idx
                            selected_cluster = None

                            if method == METHOD_GLOBAL:
                                pool = global_pool
                            elif method == METHOD_CLUSTER_TARGET:
                                selected_cluster, pool = route_to_cluster_pool(
                                    base,
                                    row_dict,
                                    target_centers,
                                    target_store,
                                    target_log,
                                    target_spec,
                                )
                            elif method == METHOD_C3_SAMPLE_DP_LLM:
                                selected_cluster, pool = route_to_cluster_pool_dp_centroid_fallback(
                                    base,
                                    row_dict,
                                    c3_centers,
                                    c3_store,
                                    c3_log,
                                    c3_spec,
                                )
                            else:
                                raise AssertionError(method)

                            shots = base.sample_balanced_shots(
                                pool,
                                k=base.SHOT_COUNTS[0],
                                seed=row_seed,
                                label_col="target",
                            )
                            pred = base.execute_classification_strategy(
                                cfg,
                                row_dict,
                                shots,
                                num_cols + cat_cols,
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

                        result = {
                            "dataset": dataset_name,
                            "method": method,
                            "epsilon": float(epsilon),
                            "delta": float(delta),
                            "K": int(k),
                            "seed": int(trial_seed),
                            "accuracy": base.accuracy_score(y_true, predictions),
                            "f1": base.f1_score(
                                y_true, predictions, zero_division=0
                            ),
                            "precision": base.precision_score(
                                y_true, predictions, zero_division=0
                            ),
                            "recall": base.recall_score(
                                y_true, predictions, zero_division=0
                            ),
                            "uses_dp_summaries": method == METHOD_C3_SAMPLE_DP_LLM,
                            "weight_source": (
                                "dp_summary_qwen_disjoint_c3"
                                if method == METHOD_C3_SAMPLE_DP_LLM
                                else (
                                    "public_schema_task_semantics"
                                    if method == METHOD_CLUSTER_TARGET else "none"
                                )
                            ),
                            "encoding": (
                                "weighted_onehot_diffprivlib_dp_kmeans"
                                if method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM}
                                else "none"
                            ),
                            "eps_cluster": (
                                float(eps_cluster)
                                if method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM}
                                else 0.0
                            ),
                            "delta_cluster": (
                                float(delta_cluster)
                                if method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM}
                                else 0.0
                            ),
                            "delta_synth": (
                                float(delta_synth)
                                if method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM}
                                else float(delta)
                            ),
                            "eps_synth": (
                                float(eps_synth)
                                if method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM}
                                else float(epsilon)
                            ),
                            "target_centered": method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM},
                            "base_synthesizer": "GEM",
                            "clusterer": (
                                "diffprivlib_weighted_onehot_dp_kmeans"
                                if method in {METHOD_CLUSTER_TARGET, METHOD_C3_SAMPLE_DP_LLM}
                                else "none"
                            ),
                            "fallback_policy": (
                                "dp_centroid_prototype_pool"
                                if method == METHOD_C3_SAMPLE_DP_LLM
                                else (
                                    "nearest_valid_cluster_reroute"
                                    if method == METHOD_CLUSTER_TARGET
                                    else "none"
                                )
                            ),
                            "n_bins": int(n_bins),
                            "fallback_clusters": (
                                int(sum(v.get("fallback", False) for v in target_log.values()))
                                if method == METHOD_CLUSTER_TARGET
                                else (
                                    int(sum(v.get("fallback", False) for v in c3_log.values()))
                                    if method == METHOD_C3_SAMPLE_DP_LLM else 0
                                )
                            ),
                            "weight_sample_fraction": (
                                float(weight_sample_frac)
                                if method == METHOD_C3_SAMPLE_DP_LLM else 0.0
                            ),
                            "weight_split_seed": (
                                int(weight_split_seed)
                                if method == METHOD_C3_SAMPLE_DP_LLM else None
                            ),
                            "weight_cache_path": (
                                c3_weight_cache_path
                                if method == METHOD_C3_SAMPLE_DP_LLM else None
                            ),
                            "main_partition_rows": (
                                int(len(c3_main_df))
                                if method == METHOD_C3_SAMPLE_DP_LLM else int(len(train_df))
                            ),
                        }
                        base.append_row_to_csv(result, results_csv)

                        retrieval_path = str(Path(results_csv).with_suffix("")) + "_retrieval.csv"
                        for item in retrieval_rows:
                            item.update(
                                dataset=dataset_name,
                                method=method,
                                epsilon=float(epsilon),
                                K=int(k),
                                seed=int(trial_seed),
                            )
                            base.append_row_to_csv(item, retrieval_path)

                        print(
                            f"[RESULT] {dataset_name} {method} "
                            f"eps={epsilon} K={k} seed={trial_seed} "
                            f"F1={result['f1']:.4f}"
                        )

                    base.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GLOBAL_GEM and C3_GEM_ICL_DP_SAMPLE_LLM using the current base pipeline."
    )
    parser.add_argument(
        "--base-script",
        default=None,
        help="Path to the unchanged PriTabICL Python script.",
    )
    parser.add_argument(
        "--gem-home",
        default=None,
        help="Path to the official terranceliu/iterative-dp repository. "
             "Defaults to GEM_HOME or /content/iterative-dp.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Adult", "Airline", "Diabetes", "Phishing"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(ALL_METHODS),
        default=list(ALL_METHODS),
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--k-list", nargs="+", type=int, default=[3])
    parser.add_argument("--eps-list", nargs="+", type=float, default=[2.0])
    parser.add_argument("--results-csv", default="gem_pritabicl_results.csv")
    parser.add_argument(
        "--cluster-share",
        type=float,
        default=0.30,
        help="Fraction of total epsilon allocated to weighted DP clustering.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="GEM delta. The base DP-KMeans clustering is pure epsilon-DP and consumes no delta.",
    )
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument(
        "--gem-kwargs-json",
        default="{}",
        help="Extra keyword arguments passed to build_gem_pool as JSON.",
    )
    parser.add_argument(
        "--weight-sample-frac",
        type=float,
        default=0.10,
        help="Disjoint fraction reserved for C3 DP-summary/Qwen weights.",
    )
    parser.add_argument("--weight-split-seed", type=int, default=314159)
    parser.add_argument("--weight-summary-base-seed", type=int, default=271828)
    parser.add_argument(
        "--hybrid-weight-cache-dir",
        default="qwen_dp_weights_cache",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        gem_kwargs = json.loads(args.gem_kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--gem-kwargs-json must be valid JSON") from exc
    if not isinstance(gem_kwargs, dict):
        raise ValueError("--gem-kwargs-json must decode to a JSON object")

    base_script = find_base_script(args.base_script)
    print(f"[BASE SCRIPT] {base_script}")
    base = load_base_module(base_script)

    gem_home = find_gem_home(args.gem_home)
    print(f"[GEM HOME] {gem_home}")
    gem_components = load_official_gem(gem_home)

    run_experiment(
        base=base,
        gem_components=gem_components,
        datasets=args.datasets,
        methods=args.methods,
        trials=args.trials,
        k_list=args.k_list,
        eps_list=args.eps_list,
        results_csv=args.results_csv,
        epsilon_cluster_share=args.cluster_share,
        delta=args.delta,
        n_bins=args.n_bins,
        gem_kwargs=gem_kwargs,
        weight_sample_frac=args.weight_sample_frac,
        weight_split_seed=args.weight_split_seed,
        weight_summary_base_seed=args.weight_summary_base_seed,
        hybrid_weight_cache_dir=args.hybrid_weight_cache_dir,
    )


if __name__ == "__main__":
    main()
