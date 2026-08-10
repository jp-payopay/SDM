from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VIFStep:
    step: int
    vifs: dict[str, float]
    dropped: str | None

    def as_dict(self) -> dict:
        return {"step": self.step, "vifs": self.vifs, "dropped": self.dropped}


@dataclass
class VIFReport:
    cutoff: float
    steps: list[VIFStep] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cutoff": self.cutoff,
            "retained": self.retained,
            "dropped": self.dropped,
            "steps": [s.as_dict() for s in self.steps],
        }


def stepwise_vif(
    X: np.ndarray,
    feature_names: list[str],
    cutoff: float = 10.0,
) -> tuple[np.ndarray, list[str], VIFReport]:
    """Drop the highest-VIF feature until all VIFs are below `cutoff`.

    Returns (X_reduced, kept_names, report). Rows with any NaN are ignored
    when computing VIFs but the returned X keeps the full row set (columns
    dropped only).
    """
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X columns.")
    finite_mask = np.all(np.isfinite(X), axis=1)
    working = X[finite_mask]
    names = list(feature_names)
    step = 0
    report = VIFReport(cutoff=cutoff)
    dropped: list[str] = []

    while working.shape[1] >= 2:
        vifs = _compute_vifs(working)
        vif_map = {names[i]: float(vifs[i]) for i in range(len(names))}
        max_i = int(np.argmax(vifs))
        max_v = float(vifs[max_i])
        if max_v <= cutoff or not np.isfinite(max_v):
            report.steps.append(VIFStep(step=step, vifs=vif_map, dropped=None))
            break
        drop_name = names[max_i]
        report.steps.append(VIFStep(step=step, vifs=vif_map, dropped=drop_name))
        dropped.append(drop_name)
        working = np.delete(working, max_i, axis=1)
        names.pop(max_i)
        step += 1

    kept_idx = [feature_names.index(n) for n in names]
    X_reduced = X[:, kept_idx]
    report.retained = names
    report.dropped = dropped
    return X_reduced, names, report


def _compute_vifs(X: np.ndarray) -> np.ndarray:
    """VIF_i = 1 / (1 - R_i^2), where R_i^2 comes from OLS of column i on the rest."""
    n, p = X.shape
    vifs = np.zeros(p)
    Xc = X - X.mean(axis=0, keepdims=True)
    for i in range(p):
        y = Xc[:, i]
        others = np.delete(Xc, i, axis=1)
        # Add intercept implicitly by centering; solve least squares.
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        y_hat = others @ coef
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float((y ** 2).sum())
        if ss_tot <= 0:
            vifs[i] = np.inf
            continue
        r2 = 1.0 - ss_res / ss_tot
        r2 = min(max(r2, 0.0), 0.999999)
        vifs[i] = 1.0 / (1.0 - r2)
    return vifs
