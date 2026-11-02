import numpy as np
import pandas as pd

from openpredico.data.entsoe_bids import ladder, read_xml


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


def _bid(status: str, direction: str, qty: int, price: float) -> str:
    return f"""<Bid_TimeSeries><status><value>{status}</value></status>
    <flowDirection.direction>{direction}</flowDirection.direction>
    <Period><timeInterval><start>2026-04-01T02:15Z</start><end>2026-04-01T02:30Z</end>
    </timeInterval><resolution>PT15M</resolution><Point><position>1</position>
    <quantity.quantity>{qty}</quantity.quantity><energy_Price.amount>{price}</energy_Price.amount>
    </Point></Period></Bid_TimeSeries>"""


def test_read_xml_keeps_available_bids():
    ns = "urn:iec62325.351:tc57wg16:451-7:reservebiddocument:7:1"
    bids = _bid("A06", "A01", 27, 160.5) + _bid("A06", "A02", 10, -5) + _bid("A11", "A01", 9, 1)
    xml = f'<ReserveBid_MarketDocument xmlns="{ns}">{bids}</ReserveBid_MarketDocument>'
    df = read_xml(xml.encode())
    assert df.direction.tolist() == ["up", "down"]
    assert df.mw.tolist() == [27, 10] and df.price.tolist() == [160.5, -5]
    assert (df.time == pd.Timestamp("2026-04-01 02:15", tz="UTC")).all()
