from __future__ import annotations

import numpy as np


def kfold(
    n: int,
    k: int = 5,
    rng: np.random.Generator | None = None,
    y: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """k-fold split. If `y` (e.g. presence/background labels) is given, folds
    are stratified — each label's indices are split into k pieces
    independently, so every fold's class balance matches the overall
    dataset instead of drifting with an unstratified shuffle (which gets
    noisier the smaller/more imbalanced the dataset is)."""
    if k < 2:
        raise ValueError("k must be >= 2")
    if rng is None:
        rng = np.random.default_rng()
    if y is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = list(np.array_split(idx, k))
    else:
        y = np.asarray(y)
        folds = [np.empty(0, dtype=int) for _ in range(k)]
        for label in np.unique(y):
            cls_idx = np.where(y == label)[0]
            rng.shuffle(cls_idx)
            for i, piece in enumerate(np.array_split(cls_idx, k)):
                folds[i] = np.concatenate([folds[i], piece])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((train, test))
    return out
