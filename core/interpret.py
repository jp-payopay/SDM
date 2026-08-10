from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Note:
    """One interpretation note for the report. `tone` drives the color of the
    callout: "good", "ok", "caution", or "info"."""

    heading: str
    tone: str
    text: str
    caveat: str = ""

    def as_dict(self) -> dict:
        return {"heading": self.heading, "tone": self.tone, "text": self.text, "caveat": self.caveat}


# The thresholds below are common rule-of-thumb conventions, not hard rules.
# They are kept in one place so they are easy to review or adjust.

def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _fmt(v) -> str:
    return f"{v:.3f}" if _finite(v) else "n/a"


def _auc_band(v: float) -> tuple[str, str]:
    if v >= 0.9:
        return "excellent", "good"
    if v >= 0.8:
        return "good", "good"
    if v >= 0.7:
        return "acceptable", "ok"
    return "low", "caution"


def _boyce_band(v: float) -> tuple[str, str]:
    if v >= 0.8:
        return "strong", "good"
    if v >= 0.5:
        return "moderate", "ok"
    if v >= 0.0:
        return "weak", "caution"
    return "counter-predicting", "caution"


def build_interpretation(
    *,
    metrics_summary: list[dict],
    n_presence: int | None,
    split_method: str,
    uncertainty_hi_frac: float | None = None,
    extrap_frac: float | None = None,
    has_maxent: bool = False,
    presence_only: bool = False,
    suitable_frac: float | None = None,
    ensemble_train_auc: float | None = None,
    ensemble_cv_auc: float | None = None,
    vif: dict | None = None,
    cleaning_removed_frac: float | None = None,
) -> list[Note]:
    """Deterministic, rule-based narrative notes derived only from values the
    run already computed. Each note interprets what a result implies for this
    run rather than defining what the metric is, and several appear only when
    they flag something. Scoped to defensible, generalizable results; it never
    guesses at the ecological meaning of a variable or a location.
    """
    notes: list[Note] = []
    if not metrics_summary:
        return notes

    # The headline model is the ensemble (what you actually deploy) if present,
    # else the best individual algorithm by mean TSS.
    headline = next((r for r in metrics_summary if r.get("algorithm") == "Ensemble"), None)
    is_ensemble = headline is not None
    if headline is None:
        headline = max(
            metrics_summary,
            key=lambda r: r["tss_mean"] if _finite(r.get("tss_mean")) else float("-inf"),
        )
    who = "ensemble" if is_ensemble else f"best model ({headline.get('algorithm', 'model')})"
    auc = headline.get("auc_mean")
    tss = headline.get("tss_mean")
    boyce = headline.get("boyce_mean")

    # ---- Overall performance (narrative synthesis of the three metrics) ----
    nums = f"(AUC {_fmt(auc)}, TSS {_fmt(tss)}, Boyce {_fmt(boyce)})"
    algo_rows = [r for r in metrics_summary if r.get("algorithm") != "Ensemble"]
    tss_vals = [r["tss_mean"] for r in algo_rows if _finite(r.get("tss_mean"))]
    spread_txt = ""
    if len(tss_vals) >= 2:
        lo, hi = min(tss_vals), max(tss_vals)
        spread_txt = f" Individual algorithms ranged from {lo:.2f} to {hi:.2f} in TSS"
        spread_txt += ", so the choice of algorithm matters here." if (hi - lo) >= 0.2 else "."
    if _finite(tss) and tss >= 0.6:
        tone = "good"
        body = (
            f"The {who} predicts this distribution well {nums}: it separates suitable from "
            f"unsuitable conditions clearly and its suitability values track where the species "
            f"actually occurs, so the map is dependable for this study area."
        )
    elif _finite(tss) and tss >= 0.4:
        tone = "ok"
        body = (
            f"The {who} captures a real but modest signal {nums}: it picks out broad suitable "
            f"areas while blurring finer distinctions. Use the map for general patterns rather "
            f"than site-level decisions."
        )
    else:
        tone = "caution"
        body = (
            f"The {who} performs weakly on this data {nums}: it barely distinguishes occupied "
            f"sites from the wider background, so treat the map as an exploratory first pass "
            f"rather than a firm prediction."
        )
    notes.append(Note("Overall performance", tone, body + spread_txt))

    # ---- Cross-metric consistency (only when the metrics disagree) ----
    if _finite(auc) and _finite(boyce):
        if _auc_band(auc)[1] in ("good", "ok") and _boyce_band(boyce)[1] == "caution":
            notes.append(Note(
                "A mixed signal", "caution",
                f"The model ranks sites better than its suitability surface reflects reality "
                f"(AUC {_fmt(auc)} but Boyce {_fmt(boyce)}). Its ordering of good versus poor "
                f"sites is more trustworthy than the raw suitability values, so favour the "
                f"thresholded (binary) map over the continuous gradient.",
            ))

    # ---- Overfitting (train vs cross-validated, only when notable) ----
    if _finite(ensemble_train_auc) and _finite(ensemble_cv_auc):
        gap = ensemble_train_auc - ensemble_cv_auc
        if gap >= 0.15:
            notes.append(Note(
                "Overfitting", "caution",
                f"The models score far higher on their training data than under cross-validation "
                f"(AUC {ensemble_train_auc:.2f} versus {ensemble_cv_auc:.2f}), a sign they have fit "
                f"detail specific to your sample. Expect the map to look sharper and more confident "
                f"than the species' real requirements justify.",
            ))
        elif gap >= 0.08:
            notes.append(Note(
                "Some overfitting", "info",
                f"Training performance runs ahead of cross-validated performance "
                f"(AUC {ensemble_train_auc:.2f} versus {ensemble_cv_auc:.2f}), so the models are "
                f"picking up some sample-specific detail. Read fine structure in the map with a "
                f"little caution.",
            ))
        elif gap < 0.05 and ensemble_cv_auc >= 0.70:
            notes.append(Note(
                "Generalization", "good",
                f"Training and cross-validated performance are close "
                f"(AUC {ensemble_train_auc:.2f} versus {ensemble_cv_auc:.2f}), so the model is "
                f"learning a transferable pattern rather than memorizing your points.",
            ))

    # ---- Run-to-run stability ----
    sds = [headline.get(k) for k in ("auc_sd", "tss_sd", "boyce_sd")]
    sds = [s for s in sds if _finite(s)]
    if sds:
        worst = max(sds)
        if worst > 0.10:
            notes.append(Note(
                "Run-to-run stability", "caution",
                f"Scores swing noticeably between replicates (largest SD {worst:.3f}), so any single "
                f"run could look better or worse than this. Don't over-read small differences between "
                f"algorithms.",
            ))
        elif worst <= 0.05:
            notes.append(Note(
                "Run-to-run stability", "good",
                f"Scores barely move between replicates (largest SD {worst:.3f}), so this result is "
                f"robust to the random split.",
            ))
        else:
            notes.append(Note(
                "Run-to-run stability", "ok",
                f"Scores wobble modestly between replicates (largest SD {worst:.3f}); read them as "
                f"approximate rather than exact.",
            ))

    # ---- Agreement between algorithms ----
    if uncertainty_hi_frac is not None:
        pct = uncertainty_hi_frac * 100.0
        if pct < 10:
            notes.append(Note(
                "Agreement between algorithms", "good",
                f"The algorithms tell a consistent story almost everywhere, diverging over only "
                f"{pct:.1f}% of the map.",
            ))
        elif pct >= 25:
            notes.append(Note(
                "Agreement between algorithms", "caution",
                f"The algorithms disagree over a substantial {pct:.1f}% of the map, so much of the "
                f"prediction depends on which model you trust. Lean on the uncertainty layer in those "
                f"areas.",
            ))
        else:
            notes.append(Note(
                "Agreement between algorithms", "info",
                f"The algorithms diverge over {pct:.1f}% of the map; check the uncertainty layer "
                f"before trusting those areas.",
            ))

    # ---- Predicted suitable footprint ----
    if suitable_frac is not None:
        pct = suitable_frac * 100.0
        if pct < 10:
            qualifier = "a fairly restricted footprint"
        elif pct > 50:
            qualifier = "a broad footprint, so the suitable/unsuitable boundary is drawn loosely"
        else:
            qualifier = "a moderate share of the landscape"
        notes.append(Note(
            "Predicted suitable area", "info",
            f"At the max-TSS threshold the {who} marks {pct:.1f}% of the study area as suitable, "
            f"{qualifier}.",
        ))

    # ---- Extrapolation in the projection ----
    if extrap_frac is not None:
        pct = extrap_frac * 100.0
        if pct >= 20:
            notes.append(Note(
                "Extrapolation in the projection", "caution",
                f"A sizeable {pct:.1f}% of the projection sits in conditions the model never saw in "
                f"training, so predictions across much of it are extrapolation and should be read as "
                f"tentative. The MOP layer shows how far outside they fall.",
            ))
        else:
            notes.append(Note(
                "Extrapolation in the projection", "info",
                f"{pct:.1f}% of the projection falls in conditions outside the training range, so "
                f"predictions there lean on extrapolation and carry extra uncertainty.",
            ))

    # ---- How performance was tested ----
    if split_method == "spatial_block":
        notes.append(Note(
            "How performance was tested", "good",
            "Because the data were split into spatial blocks, these scores reflect predicting into "
            "genuinely new ground rather than filling gaps between nearby points, which is a stricter "
            "and more honest test than a random split.",
        ))
    else:
        pretty = split_method.replace("_", " ")
        notes.append(Note(
            "How performance was tested", "ok",
            f"You used {pretty} cross-validation, which can flatter a model when occurrence points "
            f"cluster in space. Spatial block CV would give a tougher, more realistic read.",
        ))

    # ---- Sample size ----
    if n_presence is not None:
        if n_presence < 25:
            tone = "caution"
            msg = (f"With only {n_presence} presence records this is a small dataset, so every result "
                   f"here is provisional; the models have little to learn from and overfit easily.")
        elif n_presence < 50:
            tone = "caution"
            msg = (f"At {n_presence} presence records the dataset is on the small side; treat the "
                   f"results as indicative rather than settled.")
        elif n_presence < 100:
            tone = "ok"
            msg = (f"{n_presence} presence records is a workable but modest sample, enough for broad "
                   f"patterns but thin for fine detail.")
        else:
            tone = "info"
            msg = f"{n_presence} presence records is a solid sample for the algorithms used."
        if has_maxent:
            msg += " MaxEnt automatically simplified its feature set to match this sample size."
        notes.append(Note("Sample size", tone, msg))

    # ---- Dropped (collinear) predictors, only when some were dropped ----
    if vif:
        dropped = vif.get("dropped") or []
        if dropped:
            shown = ", ".join(dropped[:6]) + ("..." if len(dropped) > 6 else "")
            notes.append(Note(
                "Dropped predictors", "info",
                f"{len(dropped)} predictor(s) were set aside during selection for overlapping too "
                f"strongly with others ({shown}). They aren't necessarily unimportant ecologically; "
                f"the model kept one of each correlated group to avoid double-counting.",
            ))

    # ---- Presence-only framing ----
    if presence_only:
        notes.append(Note(
            "What the scores mean here", "info",
            "You modeled from presence records against sampled background rather than true absences, "
            "so these scores describe how well the model separates your occurrences from the wider "
            "landscape. Read them as relative skill, not literal accuracy.",
        ))

    # ---- Records removed in cleaning, only when a meaningful share was lost ----
    if cleaning_removed_frac is not None and cleaning_removed_frac >= 0.10:
        pct = cleaning_removed_frac * 100.0
        if cleaning_removed_frac >= 0.30:
            notes.append(Note(
                "Records removed in cleaning", "caution",
                f"Cleaning and thinning removed {pct:.0f}% of your input records, a large cut. Confirm "
                f"those were genuine duplicates or errors rather than good data lost to coordinate "
                f"problems.",
            ))
        else:
            notes.append(Note(
                "Records removed in cleaning", "info",
                f"Cleaning and thinning removed {pct:.0f}% of your input records, which is normal "
                f"housekeeping but worth a glance.",
            ))

    return notes
