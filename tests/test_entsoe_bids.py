import numpy as np
import pandas as pd

from openpredico.data.entsoe_bids import ladder


def test_ladder_walks_merit_order_per_direction():
    t = pd.Timestamp("2026-01-01", tz="UTC")
    bids = pd.DataFrame(
        {
            "time": [t] * 5,
            "direction": ["up", "up", "up", "down", "down"],
            "price": [90.0, 50.0, 70.0, 10.0, 30.0],
            "mw": [100.0, 40.0, 30.0, 60.0, 80.0],
        }
    )
    row = ladder(bids).loc[t]
    assert row.bid_up_p0 == 50 and row.bid_up_p50 == 70 and row.bid_up_p100 == 90
    assert row.bid_up_mw == 170 and np.isnan(row.bid_up_p200)
    assert row.bid_down_p0 == 30 and row.bid_down_p100 == 10 and np.isnan(row.bid_down_p200)
    assert row.bid_mid_p100 == 50
