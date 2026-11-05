import pytest

from openpredico.config import load_targets


def test_repo_targets_are_valid():
    targets = load_targets()
    assert {"spain_mfrr_energy", "spain_mfrr_price"} <= targets.keys()


def test_invalid_target_is_rejected(tmp_path):
    bad = tmp_path / "targets.toml"
    bad.write_text(
        '[x]\npredico_resource = "x"\ntruth = "esios:1"\nunit = "MW"\nkind = "generation"\n'
        'timezone = "Europe/Madrid"\norigin = "13:00"\ngate = "17:00"\n'
        'test_period = ["2025-11-01", "2026-10-31"]\n'
    )
    with pytest.raises(ValueError, match="capacity"):
        load_targets(bad)
