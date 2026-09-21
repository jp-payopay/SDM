from __future__ import annotations

from pathlib import Path

import numpy as np

from . import PUBLICATION_DPI


def importance_summary(
    per_replicate_importance: list[dict[str, float]],
) -> list[tuple[str, float, float, int]]:
    """(feature, mean, sd, n) across replicates, ascending by mean — exactly
    the bars and error bars `plot_variable_importance` draws, bottom to top.

    A replicate that never scored a feature contributes 0.0 for it rather than
    being dropped, so every feature's mean is over the same denominator.
    """
    features = sorted({k for rep in per_replicate_importance for k in rep})
    rows = [
        (
            f,
            float(np.mean([r.get(f, 0.0) for r in per_replicate_importance])),
            float(np.std([r.get(f, 0.0) for r in per_replicate_importance])),
            len(per_replicate_importance),
        )
        for f in features
    ]
    return sorted(rows, key=lambda row: row[1])


def plot_variable_importance(
    *,
    algorithm: str,
    per_replicate_importance: list[dict[str, float]],
    out_dir: str | Path,
) -> Path:
    """Bar plot of mean permutation importance ± SD across replicates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = importance_summary(per_replicate_importance)
    features_o = [row[0] for row in summary]
    means_o = np.array([row[1] for row in summary])
    sds_o = np.array([row[2] for row in summary])

    fig, ax = plt.subplots(figsize=(5, max(2.5, 0.35 * len(features_o))), dpi=110)
    y_pos = np.arange(len(features_o))
    ax.barh(y_pos, means_o, xerr=sds_o, color="C1", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_o)
    ax.set_xlabel("Permutation importance (Δ AUC)")
    ax.set_title(f"{algorithm}: variable importance")
    ax.axvline(0.0, color="black", linewidth=0.5)
    fig.tight_layout()
    fname = out_dir / f"importance_{algorithm}.png"
    fig.savefig(fname, dpi=PUBLICATION_DPI)
    plt.close(fig)
    return fname
