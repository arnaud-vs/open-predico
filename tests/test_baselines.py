import numpy as np
import pandas as pd

from openpredico.models.baselines import slot_climatology


def test_slot_climatology_shape_order_and_dst():
    idx = pd.date_range("2026-09-01", "2026-10-24", freq="15min", tz="UTC", inclusive="left")
    slot = idx.tz_convert("Europe/Madrid").hour * 4 + idx.tz_convert("Europe/Madrid").minute // 15
    history = pd.Series(slot + np.random.default_rng(0).normal(0, 1, len(idx)), index=idx)
    days = pd.date_range("2026-10-25", periods=2, tz="Europe/Madrid")  # 25 Oct: DST ends
    fc = slot_climatology(history, days)
    assert len(fc) == 100 + 96
    assert fc.notna().all().all()
    assert (fc.q10 <= fc.q50).all() and (fc.q50 <= fc.q90).all()
    noon = fc.tz_convert("Europe/Madrid").at_time("12:00")
    assert (noon.q50.round() == 48).all()  # slot pattern carried to the right local time
