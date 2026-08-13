from sdm_plugin.core.config import SDMConfig


def test_config_defaults_validate_incomplete():
    cfg = SDMConfig()
    errors = cfg.validate()
    assert any("Occurrence" in e for e in errors)
    assert any("predictor raster" in e for e in errors)
    assert any("Output directory" in e for e in errors)


def _valid_base_config(tmp_path):
    cfg = SDMConfig()
    cfg.occurrence.path = "/data/occ.csv"
    cfg.rasters.paths = ["/data/a.tif"]
    cfg.output.directory = str(tmp_path)
    return cfg


def test_validate_rejects_out_of_range_test_size(tmp_path):
    cfg = _valid_base_config(tmp_path)
    cfg.split.method = "random"
    cfg.split.test_size = 0.0
    errors = cfg.validate()
    assert any("test_size" in e for e in errors)

    cfg.split.test_size = 1.0
    errors = cfg.validate()
    assert any("test_size" in e for e in errors)

    cfg.split.test_size = 0.25
    errors = cfg.validate()
    assert not any("test_size" in e for e in errors)


def test_validate_rejects_enfa_with_presence_absence(tmp_path):
    cfg = _valid_base_config(tmp_path)
    cfg.data_mode = "presence_absence"
    cfg.occurrence.presence_field = "presence"
    cfg.modeling.algorithms = ["enfa"]
    errors = cfg.validate()
    assert any("ENFA" in e for e in errors)

    cfg.data_mode = "presence_only"
    errors = cfg.validate()
    assert not any("ENFA" in e for e in errors)


def test_validate_rejects_non_positive_manual_block_size(tmp_path):
    cfg = _valid_base_config(tmp_path)
    cfg.split.method = "spatial_block"
    cfg.split.auto_block_size = False
    cfg.split.block_size = 0.0
    errors = cfg.validate()
    assert any("Block size" in e for e in errors)

    cfg.split.block_size = 5000.0
    errors = cfg.validate()
    assert not any("Block size" in e for e in errors)

    cfg.split.auto_block_size = True
    cfg.split.block_size = 0.0
    errors = cfg.validate()
    assert not any("Block size" in e for e in errors)


def test_config_roundtrip(tmp_path):
    cfg = SDMConfig()
    cfg.occurrence.path = "/data/occ.csv"
    cfg.rasters.paths = ["/data/a.tif", "/data/b.tif"]
    cfg.output.directory = str(tmp_path)
    cfg.background.count = 5000
    cfg.modeling.algorithms = ["lr", "rf"]
    cfg.modeling.replicates = 3

    p = tmp_path / "run_config.json"
    cfg.to_json(p)
    loaded = SDMConfig.from_json(p)

    assert loaded.background.count == 5000
    assert loaded.modeling.algorithms == ["lr", "rf"]
    assert loaded.modeling.replicates == 3
    assert loaded.rasters.paths == ["/data/a.tif", "/data/b.tif"]
