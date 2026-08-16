from hydra import compose, initialize
from omegaconf import OmegaConf


def test_config_composes():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        assert cfg.device == "auto"
        assert cfg.material.name == "glass_beads"
        assert cfg.geometry.name == "box"


def test_material_values_physical():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        m = cfg.material
        assert m.density > 0
        assert m.radius > 0
        assert 0.0 <= m.restitution <= 1.0
        assert 0.0 < m.poisson_ratio < 0.5
        assert m.friction_particle >= 0.0


def test_command_line_override():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["device=cpu", "run.name=test"])
        assert cfg.device == "cpu"
        assert cfg.run.name == "test"


def test_config_is_serialisable():
    """Every run must be reproducible from its dumped YAML."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        text = OmegaConf.to_yaml(cfg)
        assert OmegaConf.create(text).material.density == cfg.material.density

        
def test_material_values_physical():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        m = cfg.material
        c = m.contact
        assert m.density > 0
        assert m.radius > 0
        assert 0.0 <= c.restitution <= 1.0
        assert 0.0 < c.poisson_ratio < 0.5
        assert c.friction_particle >= 0.0
        assert c.youngs_modulus <= c.youngs_modulus_true, \
            "Scaled stiffness must not exceed the true value"


def test_all_materials_load():
    for name in ["glass_beads", "tablet_placebo", "tablet_coated"]:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config", overrides=[f"material={name}"])
            assert cfg.material.name == name
            assert cfg.material.role in ("validation", "application")


def test_validation_material_is_spherical():
    """Phase 3 validations are only meaningful against spherical published data."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["material=glass_beads"])
        assert cfg.material.shape.type == "sphere"