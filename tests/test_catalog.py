from openpredico.data.catalog import build, clean


def test_clean_strips_html_and_trims():
    assert clean("<p>Precio &amp; volumen</p>\n<b>mFRR</b>") == "Precio & volumen mFRR"
    assert len(clean("word " * 100, limit=50)) <= 50
    assert clean(None) == ""


def test_build_marks_used_columns():
    lists = {
        "esios": [
            {"id": "600", "name": "Precio SPOT", "description": "<p>x</p>"},
            {"id": "1", "name": "Hidráulica"},
        ]
    }
    df = build(lists).set_index(["source", "id"])
    spot = df.loc[("esios", "600")]
    assert spot.used and spot.group == "da_market" and spot.availability == "future"
    assert spot.column.count("da_price_") == 6
    assert not df.loc[("esios", "1")].used
    assert df.loc[("calendar", "weekday")].used
    assert not df.loc[("entsoe", "A37")].used
