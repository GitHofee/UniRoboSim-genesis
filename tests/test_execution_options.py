"""Official initialization/storage and scene parallelization policy boundaries."""
import os

import pytest

from unirobosim import (BoxGeometrySpec, EntityKind, EntityPath, EntitySpec,
    EnvironmentSpec, LifecycleError, Pose, ValidationError, WorldSpec)
from unirobosim_genesis import GenesisAdapterConfig, create_provider
from unirobosim_genesis.runtime import scene_build_options


@pytest.mark.parametrize("kwargs", [{"performance_mode": 1},
    {"parallelization_level": True}, {"parallelization_level": -1},
    {"parallelization_level": 3}])
def test_execution_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        GenesisAdapterConfig(**kwargs)


@pytest.mark.parametrize("device,count,expected", [("cpu", 1, 0), ("cpu", 2, 0),
    ("cuda", 1, 2), ("cuda", 2, 2)])
def test_device_policy_and_environment_cleanup(monkeypatch, device, count, expected):
    monkeypatch.delenv("GS_PARA_LEVEL", raising=False)
    with pytest.raises(RuntimeError, match="build failure"):
        with scene_build_options(GenesisAdapterConfig(device=device), count) as facts:
            assert facts["parallelization_level"] == expected
            assert os.environ.get("GS_PARA_LEVEL") == ("2" if device == "cuda" else None)
            raise RuntimeError("build failure")
    assert "GS_PARA_LEVEL" not in os.environ


def test_explicit_environment_wins_and_is_preserved(monkeypatch):
    monkeypatch.setenv("GS_PARA_LEVEL", "1")
    with scene_build_options(GenesisAdapterConfig(parallelization_level=2), 2) as facts:
        assert facts["parallelization_level"] == 1
        assert facts["parallelization_source"] == "environment"
        assert os.environ["GS_PARA_LEVEL"] == "1"
    assert os.environ["GS_PARA_LEVEL"] == "1"
    monkeypatch.setenv("GS_PARA_LEVEL", "invalid")
    with pytest.raises(LifecycleError):
        with scene_build_options(GenesisAdapterConfig(), 1):
            pytest.fail("invalid native option accepted")
    assert os.environ["GS_PARA_LEVEL"] == "invalid"


def test_explicit_config_optout_and_external_change_preserved(monkeypatch):
    monkeypatch.delenv("GS_PARA_LEVEL", raising=False)
    with scene_build_options(GenesisAdapterConfig(parallelization_level=1), 1) as facts:
        assert facts["parallelization_level"] == 1
        assert facts["parallelization_source"] == "config"
        os.environ["GS_PARA_LEVEL"] = "0"
    assert os.environ["GS_PARA_LEVEL"] == "0"


@pytest.mark.engine
@pytest.mark.parametrize("explicit", [False, True])
def test_real_native_execution_modes_preserve_batch_and_lifecycle(monkeypatch, explicit):
    import genesis as gs
    monkeypatch.delenv("GS_PARA_LEVEL", raising=False)
    device = os.environ.get("GENESIS_TEST_DEVICE", "cuda")
    options = {"performance_mode": False, "parallelization_level": 1} if explicit else {}
    config = GenesisAdapterConfig(device=device, enable_cameras=False, **options)
    box = EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY,
        pose=Pose((0., 0., 1.)), box=BoxGeometrySpec(dimensions_m=(.1, .1, .1), mass_kg=1.))
    spec = WorldSpec("execution-policy", (box,), environments=EnvironmentSpec(count=2))
    with create_provider(config).open() as session:
        with session.build(spec) as world:
            facts = world.asset_provenance["physics"]["execution"]
            assert facts["performance_mode_init"] is (False if explicit else device == "cuda")
            assert facts["parallelization_level"] == (1 if explicit else 2 if device == "cuda" else 0)
            assert facts["runtime_ownership"] == "adapter"
            assert "GS_PARA_LEVEL" not in os.environ
            # Both real environments evolve, independently of the parallel policy.
            world.step(4)
            rows = world.read_rigid_body(world.resolve(box.path)).positions_m.rows()
            assert len(rows) == 2
            assert rows[0] == rows[1]
            assert rows[0][2] < 1.
            world.reset()
            rows = world.read_rigid_body(world.resolve(box.path)).positions_m.rows()
            assert rows[0] == rows[1] == (0., 0., 1.)
    assert not gs._initialized
    assert "GS_PARA_LEVEL" not in os.environ
