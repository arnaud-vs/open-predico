import pytest

from openpredico.config import load_targets
from openpredico.data.groups import load_groups


def test_repo_targets_are_valid():
    targets = load_targets()
    assert {"spain_mfrr_energy", "spain_mfrr_price"} <= targets.keys()


def test_invalid_target_is_rejected(tmp_path):
    bad = tmp_path / "targets.toml"
    bad.write_text(
        '[x]\npredico_resource = "x"\ntruth = "esios:1"\ncolumn = "x"\ngroups = []\n'
        'unit = "MW"\nkind = "generation"\n'
        'timezone = "Europe/Madrid"\norigin = "13:00"\ngate = "17:00"\n'
        'test_period = ["2025-11-01", "2026-10-31"]\n'
    )
    with pytest.raises(ValueError, match="capacity"):
        load_targets(bad)


def test_target_column_is_its_truth_in_its_groups():
    groups = load_groups()
    for name, t in load_targets().items():
        columns = {c: d for g in t["groups"] for c, d in groups[g].items()}
        assert columns[t["column"]]["source"] == t["truth"], name
