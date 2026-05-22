"""Ablation matrix runner for Stage 3.

Axes (planned):
   - subset       : FD001 / FD002 / FD003 / FD004
   - graph kind   : pearson / physical / none (recurrent-only sanity)
   - GNN kind     : gcn / gat
   - sparsifier   : threshold / top_k (for pearson only)
   - fusion       : concat / gated  (toggle in recurrent_gnn.py later)
   - seed         : {7, 42, 123}  (same as Stage 2 finalists)

For each cell, dispatch to train_stage3.run() and aggregate test_rmse / test_score
across seeds into mean +/- std.

Outputs
-------
- stage3/artifacts/ablation/<axis-tag>/<run_id>/summary.json (one per run)
- stage3/artifacts/ablation/ablation_runs.csv  (one row per run)
- stage3/artifacts/ablation/ablation_summary.csv (mean +/- std per cell)

Comparison baseline columns (Stage 2 finalists), added at report time:
   FD001 GRU   6.17 / 55.7
   FD002 BiGRU 14.50 / 1879.5
   FD003 GRU   5.37 / 46.0
   FD004 BiGRU 16.77 / 1194.2
"""

from __future__ import annotations

# TODO: imports (argparse, itertools, csv, pathlib, json, statistics, numpy)
# TODO: from stage3.train_stage3 import run as train_run, parse_args (or build args programmatically)


def build_run_matrix(args) -> list[dict]:
    """Cartesian product of the selected axes, returns a list of run configs."""
    # TODO: itertools.product(subsets, graphs, gnns, seeds, ...) -> list of dicts
    raise NotImplementedError


def run_one(cell: dict) -> dict:
    """Run a single training+eval and return its summary dict."""
    # TODO: build argparse.Namespace from cell, call train_stage3.run()
    raise NotImplementedError


def aggregate(rows: list[dict]) -> list[dict]:
    """Group by axis cell (ignoring seed), compute mean / std of rmse / score."""
    # TODO: groupby (subset, graph, gnn, ...); aggregate over seed
    raise NotImplementedError


def main():
    """CLI

    Example:
        python -m stage3.ablation \\
            --subsets FD001,FD002,FD003,FD004 \\
            --graphs pearson,physical \\
            --gnns gcn,gat \\
            --seeds 7,42,123
    """
    # TODO: argparse, build_run_matrix, loop run_one, write per-run summary,
    #       aggregate to ablation_summary.csv
    raise NotImplementedError


if __name__ == "__main__":
    main()
