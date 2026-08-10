from __future__ import annotations

from .maxent import auto_feature_types
from .registry import default_hyperparameters


def build_model_config(
    algorithms: list[str],
    n_presence: int | None = None,
    overrides: dict[str, dict] | None = None,
) -> dict:
    """Serializable hyperparameter summary for a set of algorithms. This is
    exactly what build_model() constructs each model with. Used both for the
    wizard's "Show / edit model configuration" dialog and for the copy of
    this written into every run's output directory and HTML report.

    MaxEnt's feature_types is the one hyperparameter resolved adaptively from
    the training data rather than fixed at construction (see
    maxent.auto_feature_types). If n_presence is given, it is shown resolved
    to the concrete feature classes that count of presence points implies;
    otherwise the auto-selection rule is described in words so the config is
    still meaningful before any data has been loaded.

    `overrides` holds any values the user edited by hand (keyed by algorithm
    code), applied on top of the defaults so the dialog, the report, and the
    actual run all show the same numbers.
    """
    overrides = overrides or {}
    out: dict = {}
    for algo in algorithms:
        params = default_hyperparameters(algo)
        if algo == "maxent" and params.get("feature_types") is None:
            if n_presence is not None:
                params["feature_types"] = list(auto_feature_types(n_presence))
                params["feature_types_note"] = (
                    f"auto-selected for {n_presence} presence point(s)"
                )
            else:
                params["feature_types"] = (
                    "auto: resolved from the final presence count at fit time "
                    "(under 10: linear; under 15: add quadratic; under 80: add "
                    "hinge; 80 or more: add product)"
                )
        algo_overrides = overrides.get(algo) or {}
        for key, value in algo_overrides.items():
            params[key] = value
        if algo == "maxent" and "feature_types" in algo_overrides:
            params["feature_types_note"] = "set by hand in model configuration"
        out[algo] = params
    return out


def changed_overrides(
    edited_config: dict[str, dict],
    n_presence: int | None = None,
) -> dict[str, dict]:
    """Reduce a fully edited config (algorithm code to parameter dict) down to
    just the values that actually differ from the defaults, restricted to real
    constructor arguments. Informational keys such as feature_types_note are
    dropped, so the result is safe to pass straight through to build_model().
    """
    out: dict[str, dict] = {}
    for algo, params in edited_config.items():
        try:
            defaults = build_model_config([algo], n_presence=n_presence)[algo]
        except KeyError:
            continue
        real_keys = set(default_hyperparameters(algo))
        changed: dict = {}
        for key, value in params.items():
            if key not in real_keys:
                continue
            if key in defaults and value == defaults[key]:
                continue
            changed[key] = value
        if changed:
            out[algo] = changed
    return out
