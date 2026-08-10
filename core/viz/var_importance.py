from __future__ import annotations

from pathlib import Path

import numpy as np

from . import PUBLICATION_DPI


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
    features = sorted({k for rep in per_replicate_importance for k in rep})
    means = np.array([np.mean([r.get(f, 0.0) for r in per_replicate_importance]) for f in features])
    sds = np.array([np.std([r.get(f, 0.0) for r in per_replicate_importance]) for f in features])

    order = np.argsort(means)
    features_o = [features[i] for i in order]
    means_o = means[order]
    sds_o = sds[order]

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
