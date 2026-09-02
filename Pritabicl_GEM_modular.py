#!/usr/bin/env python3
"""
PriTabICL with the GEM synthesis backend.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

import Pritabicl_pipeline as pipeline


METHOD_GLOBAL = "GLOBAL_GEM"
METHOD_C3 = "C3_GEM_ICL_DP_SAMPLE_LLM"
ALL_METHODS = (METHOD_GLOBAL, METHOD_C3)


def find_gem_home(explicit_path: Optional[str] = None) -> Path:
    candidates = []
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
        "Could not find terranceliu/iterative-dp. Pass --gem-home or set GEM_HOME.\n"
        f"Searched:\n{searched}"
    )


def load_official_gem(gem_home: Path):
    gem_home = gem_home.resolve()
    if str(gem_home) not in sys.path:
        sys.path.insert(0, str(gem_home))
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
            "Unable to import official GEM from terranceliu/iterative-dp. "
            "Install that repository and its requirements first."
        ) from exc
    return gem_module, QueryManager, util_general, get_synth_data, Dataset, Domain


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
):
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
        .fillna(0)
        .round()
        .astype(int)
        .clip(0, 1)
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
):
    """Run official GEM with all 2-way or feature-target workloads."""
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

    eps0, _rho = util_general.get_eps0_zCDP(float(epsilon), float(delta), T, alpha=alpha)

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


def _validate_pool(pool: pd.DataFrame, ordered_cols: Sequence[str], context: str):
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
    gem_components,
    train_df,
    num_cols,
    cat_cols,
    cat_domains,
    epsilon,
    delta,
    n_synth,
    n_bins,
    target_centered,
    seed,
    gem_kwargs,
):
    ordered_cols = list(num_cols) + list(cat_cols) + ["target"]
    pool = build_gem_pool(
        gem_components=gem_components,
        train_df=train_df[ordered_cols].copy().reset_index(drop=True),
        num_cols=list(num_cols),
        cat_cols=list(cat_cols),
        cat_domains={k: list(v) for k, v in cat_domains.items()},
        epsilon=float(epsilon),
        delta=float(delta),
        n_synth=int(n_synth),
        n_bins=int(n_bins),
        target_centered=bool(target_centered),
        seed=int(seed),
        **dict(gem_kwargs),
    )
    return _validate_pool(pool, ordered_cols, "GEM")



def build_cluster_target_gem_pools(
    gem_components,
    ctx,
    main_df,
    feature_weights,
    k,
    eps_cluster,
    eps_synth,
    delta_synth,
    rows_per_cluster,
    random_state,
    n_bins,
    gem_kwargs,
):
    centers_w, labels, onehot_spec = pipeline.fit_diffprivlib_dp_kmeans_onehot_weighted(
        df=main_df,
        num_cols=ctx.num_cols,
        cat_cols=ctx.cat_cols,
        cat_domains=ctx.cat_domains,
        feature_weights=feature_weights,
        k=int(k),
        eps_cluster=float(eps_cluster),
        random_state=int(random_state),
    )

    ordered_cols = ctx.feature_cols + ["target"]
    stores = {}
    logs = {}
    labels_arr = np.asarray(labels)
    rows_per_cluster = max(1, int(rows_per_cluster))

    for cluster_id in range(len(centers_w)):
        center_w = np.asarray(centers_w[cluster_id], dtype=float)
        cluster_df = main_df[labels_arr == cluster_id].copy().reset_index(drop=True)

        center_decoded = pipeline.decode_weighted_onehot_center(center_w, onehot_spec)
        fallback = pipeline.make_c2adp_centroid_fallback(
            center=center_decoded,
            num_cols=ctx.num_cols,
            cat_cols=ctx.cat_cols,
            cat_domains=ctx.cat_domains,
            n_rows=rows_per_cluster,
        )[ordered_cols].reset_index(drop=True)

        if len(cluster_df) == 0:
            stores[cluster_id] = fallback
            logs[cluster_id] = {
                "fallback": True,
                "private_cluster_rows": 0,
                "target_centered": True,
                "error": "empty_cluster",
            }
            continue

        try:
            synth_seed = int(random_state) + 1000 + int(cluster_id)
            pipeline.set_seed(synth_seed)
            stores[cluster_id] = call_gem(
                gem_components=gem_components,
                train_df=cluster_df,
                num_cols=ctx.num_cols,
                cat_cols=ctx.cat_cols,
                cat_domains=ctx.cat_domains,
                epsilon=float(eps_synth),
                delta=float(delta_synth),
                n_synth=rows_per_cluster,
                n_bins=int(n_bins),
                target_centered=True,
                seed=synth_seed,
                gem_kwargs=gem_kwargs,
            )
            logs[cluster_id] = {
                "fallback": False,
                "private_cluster_rows": int(len(cluster_df)),
                "target_centered": True,
            }
        except Exception as exc:
            print(f"[GEM FALLBACK] cluster={cluster_id} error={exc!r}")
            stores[cluster_id] = fallback
            logs[cluster_id] = {
                "fallback": True,
                "private_cluster_rows": int(len(cluster_df)),
                "target_centered": True,
                "error": repr(exc),
            }

    return np.asarray(centers_w), stores, logs, onehot_spec


def make_gem_adapter(gem_components, n_bins: int, gem_kwargs: dict):
    def build_global_pool(*, ctx, epsilon, delta, n_synth, seed):
        return call_gem(
            gem_components=gem_components,
            train_df=ctx.train_df,
            num_cols=ctx.num_cols,
            cat_cols=ctx.cat_cols,
            cat_domains=ctx.cat_domains,
            epsilon=epsilon,
            delta=delta,
            n_synth=n_synth,
            n_bins=n_bins,
            target_centered=False,
            seed=seed,
            gem_kwargs=gem_kwargs,
        )

    def build_cluster_pools(
        *,
        ctx,
        main_df,
        feature_weights,
        k,
        eps_cluster,
        eps_synth,
        delta,
        rows_per_cluster,
        seed,
    ):
        return build_cluster_target_gem_pools(
            gem_components=gem_components,
            ctx=ctx,
            main_df=main_df,
            feature_weights=feature_weights,
            k=k,
            eps_cluster=eps_cluster,
            eps_synth=eps_synth,
            delta_synth=delta,
            rows_per_cluster=rows_per_cluster,
            random_state=int(seed) + 1,
            n_bins=n_bins,
            gem_kwargs=gem_kwargs,
        )

    return pipeline.BackendAdapter(
        name="GEM",
        global_method=METHOD_GLOBAL,
        c3_method=METHOD_C3,
        global_seed_offset=808,
        c3_seed_offset=1081,
        build_global_pool=build_global_pool,
        build_cluster_pools=build_cluster_pools,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run PriTabICL with GEM.")
    p.add_argument("--gem-home", default=None)
    p.add_argument("--datasets", nargs="+", default=["Adult", "Airline", "Diabetes", "Phishing"])
    p.add_argument("--methods", nargs="+", choices=list(ALL_METHODS), default=list(ALL_METHODS))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--k-list", "--k_list", nargs="+", type=int, default=[3])
    p.add_argument("--eps-list", "--eps_list", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--results-csv", "--results_csv", default="gem_results.csv")
    p.add_argument("--cluster-share", type=float, default=pipeline.PAPER_CLUSTER_SHARE)
    p.add_argument("--delta", type=float, default=pipeline.PAPER_DELTA)
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--weight-sample-frac", "--weight_sample_frac", type=float, default=pipeline.PAPER_WEIGHT_SAMPLE_FRAC)
    p.add_argument("--weight-split-seed", type=int, default=pipeline.PAPER_WEIGHT_SPLIT_SEED)
    p.add_argument("--weight-summary-base-seed", type=int, default=pipeline.PAPER_WEIGHT_SUMMARY_BASE_SEED)
    p.add_argument("--weight-cache-dir", default="qwen_dp_weights_cache")
    p.add_argument("--shots", type=int, default=pipeline.PAPER_SHOTS)
    p.add_argument("--gem-T", type=int, default=10)
    p.add_argument("--gem-alpha", type=float, default=0.5)
    p.add_argument("--gem-embedding-dim", type=int, default=128)
    p.add_argument("--gem-batch-size", type=int, default=None)
    p.add_argument("--gem-lr", type=float, default=1e-4)
    p.add_argument("--gem-max-iters", type=int, default=100)
    p.add_argument("--gem-max-idxs", type=int, default=100)
    p.add_argument("--gem-resample", action="store_true")
    p.add_argument("--gem-verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    gem_home = find_gem_home(args.gem_home)
    print(f"[GEM HOME] {gem_home}")
    gem_components = load_official_gem(gem_home)

    gem_kwargs = {
        "T": args.gem_T,
        "alpha": args.gem_alpha,
        "embedding_dim": args.gem_embedding_dim,
        "lr": args.gem_lr,
        "max_iters": args.gem_max_iters,
        "max_idxs": args.gem_max_idxs,
        "resample": args.gem_resample,
        "verbose": args.gem_verbose,
    }
    if args.gem_batch_size is not None:
        gem_kwargs["batch_size"] = int(args.gem_batch_size)

    adapter = make_gem_adapter(
        gem_components=gem_components,
        n_bins=args.n_bins,
        gem_kwargs=gem_kwargs,
    )

    pipeline.run_backend_experiments(
        adapter=adapter,
        datasets=args.datasets,
        methods=args.methods,
        trials=args.trials,
        k_list=args.k_list,
        eps_list=args.eps_list,
        results_csv=args.results_csv,
        cluster_share=args.cluster_share,
        delta=args.delta,
        weight_sample_frac=args.weight_sample_frac,
        weight_split_seed=args.weight_split_seed,
        weight_summary_base_seed=args.weight_summary_base_seed,
        weight_cache_dir=args.weight_cache_dir,
        shots=args.shots,
    )


if __name__ == "__main__":
    main()
