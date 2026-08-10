from __future__ import annotations

import numpy as np


def random_train_test(
    n: int,
    test_size: float = 0.25,
    rng: np.random.Generator | None = None,
    y: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a single (train_idx, test_idx) fold as a list. If `y` (e.g.
    presence/background labels) is given, the split is stratified — each
    label's indices are held out at `test_size` independently, so the test
    set's class balance matches the overall dataset instead of drifting with
    an unstratified shuffle (which gets noisier the smaller/more imbalanced
    the dataset is)."""
    if rng is None:
        rng = np.random.default_rng()
    if y is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        test_idx = np.sort(idx[:n_test])
        train_idx = np.sort(idx[n_test:])
    else:
        y = np.asarray(y)
        test_parts = []
        train_parts = []
        for label in np.unique(y):
            cls_idx = np.where(y == label)[0]
            rng.shuffle(cls_idx)
            n_test_cls = int(round(len(cls_idx) * test_size))
            test_parts.append(cls_idx[:n_test_cls])
            train_parts.append(cls_idx[n_test_cls:])
        test_idx = np.sort(np.concatenate(test_parts))
        train_idx = np.sort(np.concatenate(train_parts))
        if len(test_idx) == 0:
            # Every class rounded down to 0 test members (tiny dataset) —
            # still guarantee at least one held-out point overall.
            test_idx = train_idx[:1]
            train_idx = train_idx[1:]
    return [(train_idx, test_idx)]
