#!/usr/bin/env python3
"""
PriTabICL with the PrivSyn synthesis backend.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

import Pritabicl_pipeline as pipeline

METHOD_GLOBAL = "GLOBAL_PRIVSYN"
METHOD_C3 = "C3_PRIVSYN_ICL_DP_SAMPLE_LLM"
ALL_METHODS = (METHOD_GLOBAL, METHOD_C3)


def build_C3_global_privsyn_pool(train_df, num_cols, cat_cols, cat_domains, eps_total, n_synth=None, n_bins=8, delta=1e-05, privsyn_home=None, consistency_iterations=2, view_iterations=10, gum_iterations=20, label_target_only=True):
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
    df = pipeline.clip_numeric_df(df, num_cols, lo=0.0, hi=1.0)
    bin_edges = pipeline._fit_equal_width_bin_edges(df, num_cols, n_bins=n_bins, lo=0.0, hi=1.0)
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
    return synth_df


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
    centers_w, clusters, onehot_spec = pipeline.fit_diffprivlib_dp_kmeans_onehot_weighted(
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
        center_decoded = pipeline.decode_weighted_onehot_center(center_w, onehot_spec)
        fallback = pipeline.make_c2adp_centroid_fallback(
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
            pipeline.set_seed(int(rng.randint(0, 10 ** 9)))
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


def make_privsyn_adapter(
    n_bins: int = 8,
    privsyn_home: Optional[str] = None,
    consistency_iterations: int = 2,
    view_iterations: int = 10,
    gum_iterations: int = 20,
):
    def build_global_pool(*, ctx, epsilon, delta, n_synth, seed):
        pipeline.set_seed(int(seed))
        return build_C3_global_privsyn_pool(
            train_df=ctx.train_df,
            num_cols=ctx.num_cols,
            cat_cols=ctx.cat_cols,
            cat_domains=ctx.cat_domains,
            eps_total=float(epsilon),
            n_synth=int(n_synth),
            n_bins=int(n_bins),
            delta=float(delta),
            privsyn_home=privsyn_home,
            consistency_iterations=int(consistency_iterations),
            view_iterations=int(view_iterations),
            gum_iterations=int(gum_iterations),
            label_target_only=False,
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
        rng = np.random.RandomState(int(seed) + 1)
        return build_C3_privsyn_icl_all_star(
            train_df=main_df,
            num_cols=ctx.num_cols,
            cat_cols=ctx.cat_cols,
            cat_domains=ctx.cat_domains,
            eps_cluster=float(eps_cluster),
            eps_synth=float(eps_synth),
            k=int(k),
            rng=rng,
            rows_per_cluster=int(rows_per_cluster),
            feature_weights=feature_weights,
            random_state=int(seed) + 1,
            n_bins=int(n_bins),
            delta=float(delta),
            privsyn_home=privsyn_home,
            consistency_iterations=int(consistency_iterations),
            view_iterations=int(view_iterations),
            gum_iterations=int(gum_iterations),
        )

    return pipeline.BackendAdapter(
        name="PrivSyn",
        global_method=METHOD_GLOBAL,
        c3_method=METHOD_C3,
        global_seed_offset=858,
        c3_seed_offset=1081,
        build_global_pool=build_global_pool,
        build_cluster_pools=build_cluster_pools,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run PriTabICL with PrivSyn.")
    p.add_argument("--privsyn-home", default=None)
    p.add_argument("--datasets", nargs="+", default=["Adult", "Airline", "Diabetes", "Phishing"])
    p.add_argument("--methods", nargs="+", choices=list(ALL_METHODS), default=list(ALL_METHODS))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--k-list", "--k_list", nargs="+", type=int, default=[3])
    p.add_argument("--eps-list", "--eps_list", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--results-csv", "--results_csv", default="privsyn_results.csv")
    p.add_argument("--cluster-share", type=float, default=pipeline.PAPER_CLUSTER_SHARE)
    p.add_argument("--delta", type=float, default=pipeline.PAPER_DELTA)
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--weight-sample-frac", "--weight_sample_frac", type=float, default=pipeline.PAPER_WEIGHT_SAMPLE_FRAC)
    p.add_argument("--weight-split-seed", type=int, default=pipeline.PAPER_WEIGHT_SPLIT_SEED)
    p.add_argument("--weight-summary-base-seed", type=int, default=pipeline.PAPER_WEIGHT_SUMMARY_BASE_SEED)
    p.add_argument("--weight-cache-dir", default="qwen_dp_weights_cache")
    p.add_argument("--shots", type=int, default=pipeline.PAPER_SHOTS)
    p.add_argument("--consistency-iterations", type=int, default=2)
    p.add_argument("--view-iterations", type=int, default=10)
    p.add_argument("--gum-iterations", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    adapter = make_privsyn_adapter(
        n_bins=args.n_bins,
        privsyn_home=args.privsyn_home,
        consistency_iterations=args.consistency_iterations,
        view_iterations=args.view_iterations,
        gum_iterations=args.gum_iterations,
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
