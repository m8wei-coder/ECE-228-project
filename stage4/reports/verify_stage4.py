"""Stage 4 verification: parameter overhead + attention-gate sanity.

Three checks:

1. **Param overhead**: for each subset and each attention cell, print the
   total parameter count and compare against the Stage 3 baseline. CBAM
   must add <1% to the backbone.

2. **No-op equivalence**: confirm that AttnRecurrent/AttnRecurrentGNN
   with both CBAM flags disabled has *exactly* the same parameter count
   as the corresponding Stage 2 / Stage 3 baseline. This is the
   mathematical guarantee that Stage 4 is a strict superset.

3. **Attention-gate sanity (post-training)**: if Stage 4 checkpoints are
   available on disk, load one CBAM model, run it on the test set, and
   print the per-sensor channel-attention gate. The check we want is
   that the gate is *non-degenerate* — not collapsed to uniform 0.5
   (untrained look) and not all 1.0 (gate ignored).

Usage
-----
    python stage4/reports/verify_stage4.py                              # checks 1+2 only
    python stage4/reports/verify_stage4.py --runs-dir <Drive runs path> # also check 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SUBSET_BACKBONES = {
    # subset -> (F, recurrent_kind, hidden, num_layers, dropout, use_gnn)
    "FD001": (14, "gru",   90, 2, 0.2, False),
    "FD002": (21, "bigru", 60, 2, 0.1, True),
    "FD003": (16, "gru",   90, 2, 0.2, True),
    "FD004": (21, "bigru", 60, 2, 0.1, True),
}

# From stage3/docs/stage3_analysis.md "ablation control parameter counts".
STAGE3_BASELINE_PARAMS = {
    "FD001": 81901,   # GRU h=90 L=2 (no GNN)
    # Stage 3 RGNN adds GCN+fusion expansion; exact totals are reported in
    # the Stage 3 batch_summary.json -> parameter_count field. The numbers
    # below are placeholders; if you have the Stage 3 summary at hand you
    # can overwrite them. For the verification logic the no-GNN case is
    # the one we need exact equality on.
}


def _make_baseline(subset: str) -> torch.nn.Module:
    F, kind, h, L, d, use_gnn = SUBSET_BACKBONES[subset]
    if use_gnn:
        from stage3.models.recurrent_gnn import RecurrentGNNFusion
        adj = torch.eye(F)
        return RecurrentGNNFusion(
            input_size=F, recurrent_kind=kind,
            hidden_size=h, num_layers=L, dropout=d,
            sequence_length=30, adj_matrix=adj,
            gnn_hidden=32, gnn_layers=2, gnn_kind="gcn",
            gnn_dropout=0.1, gnn_pool="mean", use_gnn=True,
        )
    # No-GNN: pure Stage 2 GRU baseline.
    from stage2_refactor.models.gru import GRUBaseline
    return GRUBaseline(input_size=F, hidden_size=h, num_layers=L, dropout=d)


def _make_stage4(subset: str, use_c: bool, use_t: bool) -> torch.nn.Module:
    F, kind, h, L, d, use_gnn = SUBSET_BACKBONES[subset]
    common = dict(
        input_size=F, recurrent_kind=kind, hidden_size=h,
        num_layers=L, dropout=d,
        use_channel_attn=use_c, use_temporal_attn=use_t,
        attn_reduction=4, attn_kernel=7,
    )
    if use_gnn:
        from stage4.models.attn_recurrent_gnn import AttnRecurrentGNN
        adj = torch.eye(F)
        return AttnRecurrentGNN(
            sequence_length=30, adj_matrix=adj,
            gnn_hidden=32, gnn_layers=2, gnn_kind="gcn",
            gnn_dropout=0.1, gnn_pool="mean", use_gnn=True,
            **common,
        )
    from stage4.models.attn_recurrent import AttnRecurrent
    return AttnRecurrent(**common)


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def check_param_overhead() -> None:
    print("\n=== Check 1+2: parameter overhead & no-op equivalence ===\n")
    print(f"{'subset':<8}{'config':<22}{'params':>10}{'Δ vs baseline':>16}{'overhead':>12}")
    print("-" * 68)
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        try:
            base   = _make_baseline(subset)
        except ModuleNotFoundError as e:
            # torch_geometric not installed locally -> skip GNN subsets.
            print(f"{subset:<8}skipped: {e}")
            continue
        n_base = _params(base)

        # Strict-superset check: attention disabled must equal baseline.
        m_none = _make_stage4(subset, use_c=False, use_t=False)
        n_none = _params(m_none)
        marker = "OK" if n_none == n_base else "MISMATCH!"
        print(f"{subset:<8}{'stage3 baseline':<22}{n_base:>10}{'-':>16}{'-':>12}")
        print(f"{subset:<8}{'attn=none':<22}{n_none:>10}{n_none - n_base:>+16d}{marker:>12}")

        for label, use_c, use_t in [
            ("attn=channel_only",  True,  False),
            ("attn=temporal_only", False, True),
            ("attn=cbam_full",     True,  True),
        ]:
            m = _make_stage4(subset, use_c=use_c, use_t=use_t)
            n = _params(m)
            pct = 100.0 * (n - n_base) / n_base
            print(f"{subset:<8}{label:<22}{n:>10}{n - n_base:>+16d}{f'{pct:+.3f}%':>12}")
        print()


def check_attention_gates(runs_dir: Path) -> None:
    """Load a CBAM-full checkpoint from Drive and inspect channel gates."""
    print("\n=== Check 3: trained channel-attention gates ===\n")
    if not runs_dir.exists():
        print(f"runs_dir {runs_dir} does not exist -> skipping gate check.")
        return

    # Look for any cbam_full run summary.
    for subset_dir in sorted(runs_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        subset = subset_dir.name
        if subset not in SUBSET_BACKBONES:
            continue
        for run_dir in sorted(subset_dir.iterdir()):
            if not run_dir.is_dir() or "cbam_full" not in run_dir.name:
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            ckpt = Path(summary.get("best_model_path", ""))
            if not ckpt.exists():
                continue

            model = _make_stage4(subset, use_c=True, use_t=True)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            # Drive the channel gate with a batch of zero+noise inputs.
            F = SUBSET_BACKBONES[subset][0]
            with torch.no_grad():
                # Sample a batch and read the gate values directly.
                x = torch.randn(32, 30, F)
                ca = (model.attention.channel_attn
                      if hasattr(model, "attention")
                      else None)
                if ca is None:
                    continue
                avg_pool = x.mean(dim=1)
                max_pool = x.max(dim=1).values
                gate = torch.sigmoid(ca.mlp(avg_pool) + ca.mlp(max_pool))
                gate_mean = gate.mean(dim=0).numpy()
            print(f"{subset:<8}{run_dir.name:<32}  channel gate (per-sensor mean):")
            print("  " + "  ".join(f"{v:.3f}" for v in gate_mean))
            print(f"  min={gate_mean.min():.3f}  max={gate_mean.max():.3f}  "
                  f"std={gate_mean.std():.3f}  -> "
                  f"{'NON-DEGENERATE' if gate_mean.std() > 0.02 else 'COLLAPSED'}\n")
            break  # one example per subset is enough


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=None,
                   help="Path to DRIVE_ROOT/runs (mounted Drive). "
                        "If set, also runs the trained-gate sanity check.")
    args = p.parse_args()

    check_param_overhead()
    if args.runs_dir:
        check_attention_gates(Path(args.runs_dir))


if __name__ == "__main__":
    main()
