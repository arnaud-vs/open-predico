from openpredico.data.catalog import RULES, build, clean, first


def test_clean_strips_html_and_trims():
    assert clean("<p>Precio &amp; volumen</p>\n<b>mFRR</b>") == "Precio & volumen mFRR"
    assert len(clean("word " * 100, limit=50)) <= 50
    assert clean(None) == ""


def test_category_rules_first_match_wins():
    names = [
        "Net mFRR volume allocation",
        "Operating Hourly Program P48 France balance",  # schedule beats cross-border
        "Quarter-hourly average price secondary reserve last resort suppliers component",
        "Generation in Mallorca - Menorca",
        "Something unknown",
    ]
    assert [first(RULES, n, ["Other"])[0] for n in names] == [
        "Balancing services",
        "Schedules",
        "Settlement",
        "Non-peninsular (SNP)",
        "Other",
    ]


def test_build_marks_used_columns():
    lists = {
        "en": {"esios": [{"id": 600, "name": "Marginal price Day Ahead market"}]},
        "es": {
            "esios": [
                {"id": 600, "name": "Precio SPOT", "description": "<p>x</p>"},
                {"id": 1, "name": "Hidráulica"},
            ]
        },
        "esios_archive": [
            {
                "id": 2,
                "name": "A1_liquicomun",
                "horizon": "M",
                "archive_type": "zip",
                "taxonomy_terms": [{"name": "Downloads"}, {"name": "Settlements"}],
            }
        ],
    }
    df = build(lists).set_index(["source", "id"])
    spot = df.loc[("esios", "600")]
    assert spot.used and spot.group == "da_market" and spot.availability == "future"
    assert spot.column.count("da_price_") == 6
    assert spot["name"].startswith("Marginal") and spot.name_es == "Precio SPOT"
    assert spot.description == "x" and spot.category == "Prices & markets"
    assert df.loc[("esios", "1"), "name"] == "Hidráulica"  # Spanish fallback
    assert not df.loc[("esios", "1")].used
    assert df.loc[("calendar", "weekday")].used
    assert not df.loc[("entsoe", "A37")].used
    assert df.loc[("esios_archive", "2"), "description"] == "zip file, monthly: Settlements"
