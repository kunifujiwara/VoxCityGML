"""The package-level generate_voxcity entrypoint."""
from voxcitygml import generate_voxcity, VoxelizerConfig


def test_generate_voxcity_exported():
    assert callable(generate_voxcity)


def test_generate_voxcity_delegates_to_run(monkeypatch):
    import voxcitygml.pipeline as pl
    sentinel = object()
    monkeypatch.setattr(pl.VoxCityGML, 'run', lambda self: sentinel)
    cfg = VoxelizerConfig(citygml_path='.', center_lon=139.77,
                          center_lat=35.65, size_meters=100)
    assert generate_voxcity(cfg) is sentinel
