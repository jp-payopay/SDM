from sdm_plugin.core.interpret import build_interpretation


def _by_head(notes):
    return {n.heading: n for n in notes}


def _rows_with_ensemble(tss=0.63, auc=0.88, boyce=0.71):
    return [
        {"algorithm": "Ensemble", "auc_mean": auc, "auc_sd": 0.02, "tss_mean": tss,
         "tss_sd": 0.03, "boyce_mean": boyce, "boyce_sd": 0.05, "n_replicates": 5},
        {"algorithm": "Random Forest", "auc_mean": 0.85, "auc_sd": 0.02, "tss_mean": 0.60,
         "tss_sd": 0.04, "boyce_mean": 0.68, "boyce_sd": 0.06, "n_replicates": 5},
        {"algorithm": "ENFA", "auc_mean": 0.66, "auc_sd": 0.03, "tss_mean": 0.30,
         "tss_sd": 0.05, "boyce_mean": -0.10, "boyce_sd": 0.07, "n_replicates": 5},
    ]


def test_overall_performance_is_first_and_narrative():
    notes = build_interpretation(
        metrics_summary=_rows_with_ensemble(), n_presence=297, split_method="spatial_block",
    )
    assert notes[0].heading == "Overall performance"
    # Narrative interpretation, not a metric definition.
    assert "ensemble" in notes[0].text
    assert "measures how well" not in notes[0].text
    assert "combines sensitivity" not in notes[0].text


def test_overall_tone_tracks_tss_band():
    good = build_interpretation(metrics_summary=_rows_with_ensemble(tss=0.72),
                                n_presence=200, split_method="kfold")[0]
    ok = build_interpretation(metrics_summary=_rows_with_ensemble(tss=0.50),
                              n_presence=200, split_method="kfold")[0]
    poor = build_interpretation(metrics_summary=_rows_with_ensemble(tss=0.20),
                                n_presence=200, split_method="kfold")[0]
    assert (good.tone, ok.tone, poor.tone) == ("good", "ok", "caution")


def test_headline_falls_back_to_best_algorithm():
    rows = [r for r in _rows_with_ensemble() if r["algorithm"] != "Ensemble"]
    notes = build_interpretation(metrics_summary=rows, n_presence=297, split_method="kfold")
    assert "Random Forest" in notes[0].text  # best by TSS (0.60 > 0.30)


def test_mixed_signal_fires_when_ranking_good_but_boyce_bad():
    rows = _rows_with_ensemble(tss=0.50, auc=0.82, boyce=-0.05)
    heads = _by_head(build_interpretation(metrics_summary=rows, n_presence=200, split_method="kfold"))
    assert "A mixed signal" in heads and heads["A mixed signal"].tone == "caution"


def test_mixed_signal_absent_when_metrics_agree():
    heads = _by_head(build_interpretation(
        metrics_summary=_rows_with_ensemble(), n_presence=200, split_method="kfold"))
    assert "A mixed signal" not in heads


def test_overfitting_bands():
    rows = _rows_with_ensemble()
    over = _by_head(build_interpretation(metrics_summary=rows, n_presence=200, split_method="kfold",
                                         ensemble_train_auc=0.96, ensemble_cv_auc=0.70))
    assert over["Overfitting"].tone == "caution"
    mild = _by_head(build_interpretation(metrics_summary=rows, n_presence=200, split_method="kfold",
                                         ensemble_train_auc=0.80, ensemble_cv_auc=0.70))
    assert "Some overfitting" in mild
    gen = _by_head(build_interpretation(metrics_summary=rows, n_presence=200, split_method="kfold",
                                        ensemble_train_auc=0.86, ensemble_cv_auc=0.84))
    assert gen["Generalization"].tone == "good"


def test_suitable_area_and_agreement_and_extrapolation():
    heads = _by_head(build_interpretation(
        metrics_summary=_rows_with_ensemble(), n_presence=297, split_method="spatial_block",
        uncertainty_hi_frac=0.30, extrap_frac=0.27, suitable_frac=0.04,
    ))
    assert heads["Agreement between algorithms"].tone == "caution"       # 30% >= 25
    assert heads["Extrapolation in the projection"].tone == "caution"    # 27% >= 20
    assert "restricted footprint" in heads["Predicted suitable area"].text  # 4% < 10


def test_collinearity_presence_only_and_data_loss():
    heads = _by_head(build_interpretation(
        metrics_summary=_rows_with_ensemble(), n_presence=297, split_method="spatial_block",
        presence_only=True, vif={"retained": ["a"], "dropped": ["b", "c"]},
        cleaning_removed_frac=0.42,
    ))
    assert "Dropped predictors" in heads and "b, c" in heads["Dropped predictors"].text
    assert "What the scores mean here" in heads
    assert heads["Records removed in cleaning"].tone == "caution"        # 42% >= 30


def test_conditional_notes_absent_when_not_applicable():
    heads = _by_head(build_interpretation(
        metrics_summary=_rows_with_ensemble(), n_presence=297, split_method="spatial_block",
        presence_only=False, vif={"retained": ["a"], "dropped": []},
        cleaning_removed_frac=0.02,
    ))
    assert "Dropped predictors" not in heads
    assert "What the scores mean here" not in heads
    assert "Records removed in cleaning" not in heads


def test_small_sample_is_caution_and_mentions_maxent():
    sample = next(
        n for n in build_interpretation(
            metrics_summary=_rows_with_ensemble(), n_presence=18,
            split_method="spatial_block", has_maxent=True)
        if n.heading == "Sample size"
    )
    assert sample.tone == "caution" and "MaxEnt" in sample.text


def test_empty_metrics_yields_no_notes():
    assert build_interpretation(metrics_summary=[], n_presence=100, split_method="random") == []
