from openpredico.data import entsoe


def doc(*periods):
    """A75-style document: one TimeSeries per (start, end, resolution, {position: MW})."""
    series = "".join(
        f"<TimeSeries><Period><timeInterval><start>{s}</start><end>{e}</end></timeInterval>"
        f"<resolution>{r}</resolution>"
        + "".join(
            f"<Point><position>{k}</position><quantity>{v}</quantity></Point>"
            for k, v in pts.items()
        )
        + "</Period></TimeSeries>"
        for s, e, r, pts in periods
    )
    return f'<GL_MarketDocument xmlns="urn:x">{series}</GL_MarketDocument>'.encode()


def test_a03_carry_forward_per_series_then_sum(monkeypatch):
    xml = {
        # offshore: A03 omits positions 2-3 and 5-8 (repeat), then a gap 02:00-03:00, hourly after
        "B18": doc(
            ("2026-01-01T00:00Z", "2026-01-01T02:00Z", "PT15M", {1: 1, 4: 2}),
            ("2026-01-01T03:00Z", "2026-01-01T04:00Z", "PT60M", {1: 3}),
        ),
        # onshore: published up to 01:00 only
        "B19": doc(("2026-01-01T00:00Z", "2026-01-01T01:00Z", "PT15M", {1: 10})),
    }
    monkeypatch.setattr(entsoe, "_get", lambda document, psr, start, end: xml[psr])
    b18 = entsoe.fetch("A75_B18", "2026-01-01", "2026-01-01 05:00")
    assert list(b18.iloc[:8]) == [1, 1, 1, 2, 2, 2, 2, 2]  # A03 repeats
    assert list(b18.iloc[8:16]) == [2] * 4 + [3] * 4  # gap carried forward, hourly repeated
    assert b18.iloc[16:].isna().all()  # nothing published after 04:00: stays NaN
    wind = entsoe.fetch("A75_B18+B19", "2026-01-01", "2026-01-01 05:00")
    assert list(wind.iloc[:4]) == [11, 11, 11, 12] and wind.iloc[4:].isna().all()
