"""Build per-subset adjacency matrices for the Stage 3 GNN branch.

Three graph constructions are produced per subset:

- "pearson": |Pearson correlation| computed on the preprocessed training
             features (so the graph matches the model input distribution).
             Edges with |r| > threshold are kept; diagonal set to 0.
             Output is weighted (values in (threshold, 1]).
- "physical": hand-defined gas-path topology of the C-MAPSS engine.
              Sensors in the same component are fully connected; sensors in
              adjacent components along the gas path are also connected;
              "global" sensor 10 connects to every component; the core shaft
              connects HPC and HPT. Output is the induced subgraph on the
              retained sensors of each subset. Binary {0, 1}.
- "union": element-wise max of pearson (weighted) and physical (binary, 1.0).
           Keeps the pearson weight where available, otherwise 1.0 on
           physical-only edges. For ablation.

Colab-friendly notes
--------------------
- All paths are repo-root relative or CLI args; no absolute local paths.
- The script does NOT read reproduce/exports/ (that path is gitignored).
  Pearson is recomputed from CMaps/train_FD00X.txt via the stage2_refactor
  preprocessing pipeline, so any clone of the repo can rebuild it.
- Output .npy files (a few KB each) are written to stage3/artifacts/ and
  are whitelisted in .gitignore so they can be committed as a snapshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_refactor.data.io import read_cmapss_table
from stage2_refactor.data.preprocessing import PreprocessingParams, fit_transform_train


# Hard-coded subset configs (mirror stage3/configs/stage3.yaml subsets[*].data).
SUBSET_CONFIGS: dict[str, dict] = {
    "FD001": {
        "train_file": "CMaps/train_FD001.txt",
        "drop_cols":  ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                       "sensor_16", "sensor_18", "sensor_19"],
        "rul_window_size": 12,
        "rul_threshold":   0.2,
        "rul_patience":    1,
    },
    "FD002": {
        "train_file": "CMaps/train_FD002.txt",
        "drop_cols":  [],
        "rul_window_size": 12,
        "rul_threshold":   0.2,
        "rul_patience":    1,
    },
    "FD003": {
        "train_file": "CMaps/train_FD003.txt",
        "drop_cols":  ["sensor_1", "sensor_5",
                       "sensor_16", "sensor_18", "sensor_19"],
        "rul_window_size": 12,
        "rul_threshold":   0.2,
        "rul_patience":    2,
    },
    "FD004": {
        "train_file": "CMaps/train_FD004.txt",
        "drop_cols":  [],
        "rul_window_size": 12,
        "rul_threshold":   0.3,
        "rul_patience":    3,
    },
}


# Sensor -> engine component (per the C-MAPSS Damage Propagation Modeling
# sensor table and the Stage 3 spec).
SENSOR_COMPONENT: dict[int, str] = {
    1:  "Fan",        2:  "LPC",       3:  "HPC",       4:  "LPT",
    5:  "Fan",        6:  "Nozzle",    7:  "HPC",       8:  "Fan",
    9:  "Core",       10: "Global",    11: "HPC",       12: "Combustor",
    13: "Fan",        14: "Core",      15: "Nozzle",    16: "Combustor",
    17: "HPC",        18: "Fan",       19: "Core",      20: "HPT",
    21: "LPT",
}

# Gas-path adjacency between components. Core shaft links HPC and HPT.
# The "Global" component is handled separately (it connects to every other
# component because sensor 10 is a global / system-level signal).
COMPONENT_EDGES: set[frozenset[str]] = {
    frozenset(("Fan", "LPC")),
    frozenset(("LPC", "HPC")),
    frozenset(("HPC", "Combustor")),
    frozenset(("Combustor", "HPT")),
    frozenset(("HPT", "LPT")),
    frozenset(("LPT", "Nozzle")),
    frozenset(("Core", "HPC")),
    frozenset(("Core", "HPT")),
}


def _sensor_id(name: str) -> int:
    return int(name.split("_")[1])


def _components_adjacent(a: str, b: str) -> bool:
    if a == b:
        return True
    if a == "Global" or b == "Global":
        return True
    return frozenset((a, b)) in COMPONENT_EDGES


def _preprocess_train(subset: str) -> tuple["pd.DataFrame", list[str]]:  # noqa: F821
    """Run stage2_refactor preprocessing on the raw train txt.

    Returns (preprocessed_df, retained_feature_names).
    """
    cfg = SUBSET_CONFIGS[subset]
    train_path = REPO_ROOT / cfg["train_file"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"Raw train file missing: {train_path}. "
            f"Stage 3 expects CMaps/ at the repo root (committed by teammate)."
        )
    df = read_cmapss_table(train_path)
    params = PreprocessingParams(
        drop_cols=cfg["drop_cols"],
        rul_window_size=cfg["rul_window_size"],
        rul_threshold=cfg["rul_threshold"],
        rul_patience=cfg["rul_patience"],
    )
    df_pp, artifact = fit_transform_train(df, params, artifact_path=None)
    return df_pp, list(artifact["features"])


def build_pearson_adj(
    subset: str, threshold: float = 0.3
) -> tuple[np.ndarray, list[str]]:
    """Weighted |Pearson r| adjacency over preprocessed train features.

    Self-loops are zeroed. To use this as a GCN adjacency, the caller can
    add an identity matrix at use site (standard GCN normalization).
    """
    df, features = _preprocess_train(subset)
    corr = df[features].corr(method="pearson").to_numpy().astype(np.float32)
    adj = np.abs(corr)
    np.fill_diagonal(adj, 0.0)
    adj[adj <= threshold] = 0.0
    adj = np.maximum(adj, adj.T)  # guard against floating-point asymmetry
    return adj, features


def build_physical_adj(subset: str) -> tuple[np.ndarray, list[str]]:
    """Binary physical-topology adjacency induced on the retained sensors.

    Edges within and between gas-path-adjacent components, plus all edges
    incident to the global sensor (sensor_10) when retained.
    """
    _, features = _preprocess_train(subset)
    n = len(features)
    adj = np.zeros((n, n), dtype=np.float32)
    comps = [SENSOR_COMPONENT[_sensor_id(f)] for f in features]
    for i in range(n):
        for j in range(i + 1, n):
            if _components_adjacent(comps[i], comps[j]):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj, features


def build_union_adj(
    subset: str, threshold: float = 0.3
) -> tuple[np.ndarray, list[str]]:
    """Element-wise max of pearson (weighted) and physical (binary)."""
    a_p, f_p = build_pearson_adj(subset, threshold=threshold)
    a_t, f_t = build_physical_adj(subset)
    if f_p != f_t:
        raise RuntimeError(
            f"Feature lists diverged between pearson and physical for {subset}: "
            f"{f_p} vs {f_t}"
        )
    return np.maximum(a_p, a_t), f_p


def build_adjacency(
    subset: str, method: str, threshold: float = 0.3
) -> np.ndarray:
    """Public entry point.

    Args:
        subset: one of FD001 / FD002 / FD003 / FD004.
        method: "pearson" | "physical" | "union".
        threshold: |r| threshold for pearson and the pearson component of union.
    """
    if method == "pearson":
        adj, _ = build_pearson_adj(subset, threshold=threshold)
    elif method == "physical":
        adj, _ = build_physical_adj(subset)
    elif method == "union":
        adj, _ = build_union_adj(subset, threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return adj


def adj_stats(adj: np.ndarray) -> tuple[int, int, float]:
    """Return (n_nodes, n_undirected_edges, avg_degree)."""
    n = adj.shape[0]
    n_edges = int(np.sum(np.triu(adj > 0, k=1)))
    avg_deg = (2.0 * n_edges) / n if n else 0.0
    return n, n_edges, avg_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="|r| threshold for pearson edges (default 0.3).")
    parser.add_argument("--out-dir",
                        default=str(REPO_ROOT / "stage3" / "artifacts"),
                        help="Directory to write adj_{subset}_{method}.npy.")
    parser.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    parser.add_argument("--methods", default="pearson,physical,union")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in args.subsets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    rows: list[tuple[str, str, int, int, float, str]] = []
    for subset in subsets:
        for method in methods:
            adj = build_adjacency(subset, method, threshold=args.threshold)
            n, e, deg = adj_stats(adj)
            out_path = out_dir / f"adj_{subset}_{method}.npy"
            np.save(out_path, adj)
            rows.append((subset, method, n, e, deg,
                         str(out_path.relative_to(REPO_ROOT))))

    header = (
        f"{'subset':<8}{'method':<10}{'N':>4}{'edges':>8}"
        f"{'avg_deg':>10}  path"
    )
    print()
    print(header)
    print("-" * len(header))
    for subset, method, n, e, deg, path in rows:
        print(f"{subset:<8}{method:<10}{n:>4}{e:>8}{deg:>10.2f}  {path}")
    print()
    print(f"Pearson |r| threshold used: {args.threshold}")


if __name__ == "__main__":
    main()
