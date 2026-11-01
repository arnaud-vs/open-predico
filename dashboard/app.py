"""open-predico dashboard: backtest scores from the public open-predico/benchmark dataset.

Runs in the browser with stlite (Streamlit on Pyodide, see index.html), so the Space is a free
static one; `streamlit run app.py` works locally too.
"""

import io
import json
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO = "open-predico/benchmark"
HF = "https://huggingface.co"
PREFIX = "predictions/backtest/"
TZ = "Europe/Madrid"
ALPHA = 0.2

COLOR = alt.Color(
    "config:N", scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(orient="bottom")
)

st.set_page_config(page_title="open-predico", layout="wide")


def fetch(url: str) -> bytes:
    """GET `url`; in Pyodide via synchronous XHR (allowed in stlite's web worker)."""
    if sys.platform != "emscripten":
        import urllib.request

        with urllib.request.urlopen(url) as r:
            return r.read()
    from js import XMLHttpRequest

    req = XMLHttpRequest.new()
    req.open("GET", url, False)
    req.responseType = "arraybuffer"
    req.send()
    if req.status != 200:
        raise OSError(f"GET {url}: HTTP {req.status}")
    return req.response.to_py().tobytes()


@st.cache_data(ttl=3600)
def load() -> pd.DataFrame:
    """Per-step errors for every config and target, with local time columns."""
    tree = json.loads(
        fetch(f"{HF}/api/datasets/{REPO}/tree/main/{PREFIX.rstrip('/')}?recursive=true")
    )
    files = [f["path"] for f in tree if f["path"].endswith(".parquet")]
    out = []
    for f in files:
        config, target = f.removeprefix(PREFIX).removesuffix(".parquet").split("/")
        df = pd.read_parquet(io.BytesIO(fetch(f"{HF}/datasets/{REPO}/resolve/main/{f}")))
        # Same formulas as src/openpredico/evaluation/metrics.py (copied, keep in sync).
        o, q10, q50, q90 = df.obs, df.q10, df.q50, df.q90
        df["se"] = (o - q50) ** 2
        df["winkler"] = (
            (q90 - q10)
            + (2 / ALPHA) * np.maximum(0, q10 - o)
            + (2 / ALPHA) * np.maximum(0, o - q90)
        )
        df["ae"] = (o - q50).abs()
        df["covered"] = ((o >= q10) & (o <= q90)).astype(float)
        df["missing"] = o.isna()
        local = df.index.tz_convert(TZ)
        df["day"] = local.tz_localize(None).normalize()
        df["hour"] = local.hour
        df["weekday"] = local.dayofweek
        df["config"], df["target"] = config, target
        out.append(df.reset_index())
    return pd.concat(out, ignore_index=True)


def daily(steps: pd.DataFrame) -> pd.DataFrame:
    """Daily scores; a day with any missing truth is skipped, as Predico does."""
    g = steps.groupby(["target", "config", "day"])
    d = g.agg(
        se=("se", "mean"),
        winkler=("winkler", "mean"),
        coverage=("covered", "mean"),
        missing=("missing", "any"),
    )
    d["rmse"] = np.sqrt(d.pop("se"))
    return d[~d.pop("missing")].reset_index()


steps = load()
days = daily(steps)
targets = sorted(days.target.unique())
configs = sorted(days.config.unique())

st.title("open-predico")
st.caption(
    "Day-ahead quantile forecasts for [Predico](https://predico.inesctec.pt/docs/), Spain mFRR. "
    "[Code](https://github.com/arnaud-vs/open-predico) · "
    "[Data](https://huggingface.co/datasets/open-predico/benchmark)"
)
(tab,) = st.tabs(["Backtest"])

with tab:
    st.markdown(
        "Test year 1 Nov 2025 – 31 Oct 2026, forecast made at 14:00 Madrid on D-1. "
        "RMSE on q50, Winkler (α = 0.2) on q10/q90, per local day; lower is better. "
        "Days with any missing truth are skipped, like Predico."
    )

    st.subheader("Leaderboard")
    cols = st.columns(len(targets))
    for col, t in zip(cols, targets, strict=True):
        lb = (
            days[days.target == t]
            .groupby("config")
            .agg(
                rmse=("rmse", "mean"),
                winkler=("winkler", "mean"),
                coverage=("coverage", "mean"),
                days=("day", "count"),
            )
            .sort_values("rmse")
        )
        col.markdown(f"**{t}**")
        col.dataframe(
            lb.style.format({"rmse": "{:.1f}", "winkler": "{:.1f}", "coverage": "{:.0%}"}),
        )

    st.subheader("By month")
    target = st.selectbox("Target", targets, key="month_target")
    monthly = (
        days[days.target == target]
        .assign(month=lambda d: d.day.dt.to_period("M").dt.to_timestamp())
        .groupby(["config", "month"])[["rmse", "winkler"]]
        .mean()
        .reset_index()
    )
    cols = st.columns(2)
    for col, metric in zip(cols, ["rmse", "winkler"], strict=True):
        col.altair_chart(
            alt.Chart(monthly, title=metric.upper())
            .mark_line(point=True)
            .encode(
                x=alt.X("yearmonth(month):T", title=None),
                y=alt.Y(f"{metric}:Q", title=None),
                color=COLOR,
                tooltip=["config", "yearmonth(month):T", alt.Tooltip(f"{metric}:Q", format=".1f")],
            ),
        )

    st.subheader("Paired comparison")
    c1, c2 = st.columns(2)
    a = c1.selectbox("Config A", configs, index=len(configs) - 1)
    b = c2.selectbox("Config B", configs, index=0)
    rows = []
    for t in targets:
        p = days[days.target == t].pivot(index="day", columns="config", values=["rmse", "winkler"])
        both = p.dropna()
        rows.append(
            {
                "target": t,
                "days": len(both),
                "A wins RMSE": (both["rmse"][a] < both["rmse"][b]).mean(),
                "A wins Winkler": (both["winkler"][a] < both["winkler"][b]).mean(),
            }
        )
    st.dataframe(
        pd.DataFrame(rows)
        .set_index("target")
        .style.format({"A wins RMSE": "{:.0%}", "A wins Winkler": "{:.0%}"}),
    )

    st.subheader("Absolute error of q50 by hour and weekday (Madrid time)")
    target = st.selectbox("Target", targets, key="hour_target")
    sel = steps[steps.target == target]
    cols = st.columns(2)
    for col, key, title in [(cols[0], "hour", "Hour of day"), (cols[1], "weekday", "Weekday")]:
        prof = sel.groupby(["config", key]).ae.mean().reset_index()
        if key == "weekday":
            prof["weekday"] = prof.weekday.map(
                dict(enumerate("Mon Tue Wed Thu Fri Sat Sun".split()))
            )
        col.altair_chart(
            alt.Chart(prof, title=title)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    f"{key}:O",
                    title=None,
                    sort="Mon Tue Wed Thu Fri Sat Sun".split() if key == "weekday" else None,
                ),
                y=alt.Y("ae:Q", title="MAE"),
                color=COLOR,
                tooltip=["config", key, alt.Tooltip("ae:Q", format=".1f")],
            ),
        )

    st.subheader("Day viewer")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target", targets, key="day_target")
    lo, hi = steps.day.min().date(), steps.day.max().date()
    date = c2.date_input("Day (Madrid)", value=hi, min_value=lo, max_value=hi)
    view = steps[(steps.target == target) & (steps.day == pd.Timestamp(date))].copy()
    view["time"] = view.time.dt.tz_convert(TZ).dt.tz_localize(None)
    cfgs = st.multiselect("Configs", configs, default=configs)
    view = view[view.config.isin(cfgs)]
    shown = days[(days.target == target) & (days.day == pd.Timestamp(date))]
    if shown.empty:
        st.caption("Truth incomplete on this day: not scored.")
    else:
        st.caption(
            " · ".join(
                f"{r.config}: RMSE {r.rmse:.1f}, Winkler {r.winkler:.1f}"
                for r in shown.itertuples()
            )
        )
    base = alt.Chart(view).encode(x=alt.X("time:T", title=None))
    band = base.mark_area(opacity=0.2).encode(y=alt.Y("q10:Q", title=None), y2="q90:Q", color=COLOR)
    q50 = base.mark_line().encode(y="q50:Q", color=COLOR)
    obs = (
        alt.Chart(view[view.config == view.config.iloc[0]] if len(view) else view)
        .mark_line(color="gray", strokeDash=[4, 2])
        .encode(x="time:T", y="obs:Q")
    )
    st.altair_chart(band + q50 + obs)
    st.caption("Shaded: q10–q90 band; solid: q50; dashed grey: truth.")
