"""open-predico dashboard: backtest scores and target data from open-predico/benchmark (public).

Runs in the browser with stlite (Streamlit on Pyodide, see index.html), so the Space is a free
static one; `streamlit run app.py` works locally too. Pyodide is slow: every file is fetched once
and cached, and plots get aggregated data.
"""

import datetime as dt
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
BASELINE = "slot_climatology_28d"
METRICS = {"rmse": "RMSE", "winkler": "Winkler"}
WEEKDAYS = "Mon Tue Wed Thu Fri Sat Sun".split()
# Model type from the config name prefix; anything else is an ML model.
TYPES = {
    "slot_climatology": "baseline",
    "persistence": "baseline",
    "zero_inflated": "baseline",
    "chronos": "foundation model",
    "timesfm": "foundation model",
    "timestral": "foundation model",
    "t0": "foundation model",
    "toto": "foundation model",
}

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


def parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(fetch(f"{HF}/datasets/{REPO}/resolve/main/{path}")))


def model_type(config: str) -> str:
    return next((t for p, t in TYPES.items() if config.startswith(p)), "ML")


# --- Backtest data ---------------------------------------------------------------------------


@st.cache_resource(ttl=3600)
def load_steps() -> pd.DataFrame:
    """Per-step errors for every config folder under predictions/backtest/, with local time."""
    tree = json.loads(
        fetch(f"{HF}/api/datasets/{REPO}/tree/main/{PREFIX.rstrip('/')}?recursive=true")
    )
    files = [f["path"] for f in tree if f["path"].endswith(".parquet")]
    out = []
    for f in files:
        config, target = f.removeprefix(PREFIX).removesuffix(".parquet").split("/")
        df = parquet(f)
        # Same formulas as src/openpredico/evaluation/metrics.py (copied, keep in sync).
        o, q10, q50, q90 = df.obs, df.q10, df.q50, df.q90
        df["se"] = (o - q50) ** 2
        df["winkler"] = (
            (q90 - q10)
            + (2 / ALPHA) * np.maximum(0, q10 - o)
            + (2 / ALPHA) * np.maximum(0, o - q90)
        )
        df["ae"] = (o - q50).abs()
        df["covered"] = ((o >= q10) & (o <= q90)).astype(float).where(o.notna())
        local = df.index.tz_convert(TZ)
        df["day"] = local.tz_localize(None).normalize()
        df["hour"] = local.hour
        df["weekday"] = local.dayofweek
        df["config"], df["target"] = config, target.removeprefix("spain_mfrr_")
        out.append(df.reset_index())
    return pd.concat(out, ignore_index=True)


@st.cache_resource(ttl=3600)
def load_days() -> pd.DataFrame:
    """Daily scores over the quarter-hours that have truth; days with none are dropped."""
    g = load_steps().groupby(["target", "config", "day"])
    d = g.agg(
        se=("se", "mean"),
        winkler=("winkler", "mean"),
        coverage=("covered", "mean"),
        n=("se", "count"),
    )
    d["rmse"] = np.sqrt(d.pop("se"))
    return d[d.pop("n") > 0].reset_index()


def leaderboard(days: pd.DataFrame) -> pd.DataFrame:
    """One row per config: raw and baseline-normalised scores per target, aggregate and rank.

    Normalised = mean daily score / baseline's mean daily score, on days both are scored.
    Score = geometric mean of the normalised RMSE and Winkler over both targets.
    """
    cols = {}
    for t, d in days.groupby("target"):
        p = d.pivot(index="day", columns="config", values=["rmse", "winkler", "coverage"])
        for m in METRICS:
            cols[f"{t} {m}"] = p[m].mean()
            if BASELINE in p[m]:
                both = p[m].dropna(subset=[BASELINE])
                cols[f"{t} {m} rel"] = both.mean() / both[BASELINE].mean()
        cols[f"{t} coverage"] = p["coverage"].mean()
        cols[f"{t} days"] = p["rmse"].count()
    lb = pd.DataFrame(cols)
    rel = lb.filter(like=" rel")
    lb.insert(0, "score", np.exp(np.log(rel).mean(axis=1, skipna=False)))
    lb.insert(0, "type", [model_type(c) for c in lb.index])
    lb.insert(0, "rank", lb.score.rank(method="min"))
    return lb.sort_values(["score", "rank"]).rename_axis("config").reset_index()


def win_rates(days: pd.DataFrame, target: str, metric: str, order: list[str]) -> pd.DataFrame:
    """Matrix: share of commonly scored days where row config A beats column config B."""
    p = days[days.target == target].pivot(index="day", columns="config", values=metric)
    return pd.DataFrame(
        [
            [(p[a] < p[b])[p[a].notna() & p[b].notna()].mean() if a != b else np.nan for b in order]
            for a in order
        ],
        index=pd.Index(order, name="A \\ B"),
        columns=order,
    )


def win_color(v: float) -> str:
    """Red (A loses) to green (A wins); dark text so it reads in light and dark themes."""
    if pd.isna(v):
        return ""
    r, g = (255, int(80 + 175 * 2 * v)) if v < 0.5 else (int(255 - 175 * (2 * v - 1)), 255)
    return f"background-color: rgb({r},{g},110); color: black"


# --- Target data -----------------------------------------------------------------------------


@st.cache_resource
def load_target() -> pd.DataFrame:
    """v1/target.parquet in long form with local calendar columns."""
    df = parquet("v1/target.parquet")
    local = df.index.tz_convert(TZ)
    cal = pd.DataFrame(
        {
            "time": local.tz_localize(None),
            "day": local.tz_localize(None).normalize(),
            "slot": local.strftime("%H:%M"),
            "weekday": local.dayofweek,
            "month": local.tz_localize(None).to_period("M").to_timestamp(),
        }
    )
    out = []
    for col in df.columns:
        out.append(cal.assign(target=col.removeprefix("mfrr_"), value=df[col].to_numpy()))
    return pd.concat(out, ignore_index=True)


@st.cache_resource
def profile(target: str, key: str) -> pd.DataFrame:
    """Mean and q10/q90 of the target by a calendar key."""
    v = load_target()
    g = v[v.target == target].groupby(key).value
    return pd.DataFrame(
        {"mean": g.mean(), "q10": g.quantile(0.1), "q90": g.quantile(0.9)}
    ).reset_index()


def band_chart(df: pd.DataFrame, x: alt.X, title: str) -> alt.LayerChart:
    base = alt.Chart(df, title=title).encode(x=x)
    band = base.mark_area(opacity=0.25).encode(y=alt.Y("q10:Q", title=None), y2="q90:Q")
    line = base.mark_line(point=len(df) < 40).encode(
        y="mean:Q", tooltip=[x.shorthand, alt.Tooltip("mean:Q", format=".1f")]
    )
    return band + line


# --- Page ------------------------------------------------------------------------------------

st.title("open-predico")
st.caption(
    "Day-ahead quantile forecasts for [Predico](https://predico.inesctec.pt/docs/), Spain mFRR. "
    "[Code](https://github.com/arnaud-vs/open-predico) · "
    "[Data](https://huggingface.co/datasets/open-predico/benchmark)"
)
tab_bt, tab_target = st.tabs(["Backtest", "Target data"])

with tab_bt:
    steps, days = load_steps(), load_days()
    targets = sorted(days.target.unique())
    st.markdown(
        "Test year 1 Nov 2025 – 31 Oct 2026, forecast made at 14:00 Madrid on D-1, scored per "
        "local day; lower is better. Quarter-hours without truth (no mFRR activation, so no price) "
        "are left out."
    )

    st.subheader("Leaderboard")
    lb = leaderboard(days)
    order = lb.config.tolist()
    types = st.pills("Model type", sorted(lb.type.unique()), selection_mode="multi", default=None)
    absolute = st.toggle("Show absolute scores")
    shown = lb[lb.type.isin(types)] if types else lb
    keep = ["rank", "config", "type", "score"]
    for t in targets:
        keep += [f"{t} rmse rel", f"{t} winkler rel", f"{t} coverage"]
        keep += [f"{t} rmse", f"{t} winkler", f"{t} days"] if absolute else []
    ratio = "Mean daily {} / baseline's, on days both are scored; < 1 beats the baseline."
    colcfg = {
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
        "config": "Config",
        "type": "Type",
        "score": st.column_config.NumberColumn(
            "Score",
            format="%.3f",
            help="Geometric mean of the four normalised scores (RMSE and Winkler, both targets).",
        ),
    }
    for t in targets:
        colcfg |= {
            f"{t} rmse rel": st.column_config.NumberColumn(
                f"{t} RMSE", format="%.3f", help=ratio.format("RMSE of q50")
            ),
            f"{t} winkler rel": st.column_config.NumberColumn(
                f"{t} Winkler", format="%.3f", help=ratio.format("Winkler score of q10/q90")
            ),
            f"{t} coverage": st.column_config.NumberColumn(
                f"{t} coverage",
                format="percent",
                help="Share of quarter-hours inside q10–q90; ideal 80%.",
            ),
            f"{t} rmse": st.column_config.NumberColumn(f"{t} RMSE abs", format="%.1f"),
            f"{t} winkler": st.column_config.NumberColumn(f"{t} Winkler abs", format="%.1f"),
            f"{t} days": st.column_config.NumberColumn(f"{t} days", format="%d"),
        }
    st.dataframe(shown[keep], column_config=colcfg, hide_index=True)
    with st.expander("Metrics"):
        st.markdown(
            f"""
- **RMSE** (Predico's q50 score): `sqrt(mean_t (obs − q50)²)` over a local day.
- **Winkler** (Predico's q10/q90 score, α = 0.2): `mean_t [(q90 − q10) + 10·max(0, q10 − obs)
  + 10·max(0, obs − q90)]`. Rewards narrow intervals, penalises misses.
- **Coverage**: share of quarter-hours with obs inside [q10, q90]; a calibrated model gets 80%.
- **Normalised**: a config's mean daily score divided by the baseline `{BASELINE}` (= 1.0) on
  the days both are scored. **Score** is the geometric mean of the four normalised values;
  **Rank** sorts by it.
- **Type**: baseline, foundation model (pretrained, zero-shot), ML (trained here).
"""
        )

    st.subheader("Win rate")
    st.caption(
        "Cell (row A, column B): share of days scored for both where A has the lower score. "
        "Read across a row: green = A usually wins. Diagonal blank."
    )
    target = st.radio("Target", targets, horizontal=True, key="win_target")
    cols = st.columns(2)
    for col, (m, label) in zip(cols, METRICS.items(), strict=True):
        col.markdown(f"**{label}**")
        w = win_rates(days, target, m, order)
        txt = w.map(lambda v: "" if pd.isna(v) else f"{v:.0%}")  # st.dataframe shows NaN as None
        col.dataframe(txt.style.apply(lambda _, w=w: w.map(win_color), axis=None))

    st.subheader("By month")
    target = st.radio("Target", targets, horizontal=True, key="month_target")
    monthly = (
        days[days.target == target]
        .assign(month=lambda d: d.day.dt.to_period("M").dt.to_timestamp())
        .groupby(["config", "month"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    cols = st.columns(2)
    for col, (m, label) in zip(cols, METRICS.items(), strict=True):
        col.altair_chart(
            alt.Chart(monthly, title=label)
            .mark_line(point=True)
            .encode(
                x=alt.X("yearmonth(month):T", title=None),
                y=alt.Y(f"{m}:Q", title=None),
                color=COLOR,
                tooltip=["config", "yearmonth(month):T", alt.Tooltip(f"{m}:Q", format=".1f")],
            ),
        )

    st.subheader("Absolute error of q50 by hour and weekday (Madrid time)")
    target = st.radio("Target", targets, horizontal=True, key="hour_target")
    sel = steps[steps.target == target]
    cols = st.columns(2)
    for col, key, title in [(cols[0], "hour", "Hour of day"), (cols[1], "weekday", "Weekday")]:
        prof = sel.groupby(["config", key]).ae.mean().reset_index()
        if key == "weekday":
            prof["weekday"] = prof.weekday.map(dict(enumerate(WEEKDAYS)))
        col.altair_chart(
            alt.Chart(prof, title=title)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{key}:O", title=None, sort=WEEKDAYS if key == "weekday" else None),
                y=alt.Y("ae:Q", title="MAE"),
                color=COLOR,
                tooltip=["config", key, alt.Tooltip("ae:Q", format=".1f")],
            ),
        )

    st.subheader("Day viewer")
    c1, c2 = st.columns(2)
    target = c1.radio("Target", targets, horizontal=True, key="day_target")
    lo, hi = steps.day.min().date(), steps.day.max().date()
    date = c2.date_input("Day (Madrid)", value=hi, min_value=lo, max_value=hi)
    cfgs = st.multiselect("Configs", order, default=order)
    view = steps[
        (steps.target == target) & (steps.day == pd.Timestamp(date)) & steps.config.isin(cfgs)
    ].copy()
    view["time"] = view.time.dt.tz_convert(TZ).dt.tz_localize(None)
    scored = days[(days.target == target) & (days.day == pd.Timestamp(date))]
    if scored.empty:
        st.caption("Truth incomplete on this day: not scored.")
    else:
        st.caption(
            " · ".join(
                f"{r.config}: RMSE {r.rmse:.1f}, Winkler {r.winkler:.1f}"
                for r in scored.itertuples()
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

with tab_target:
    tv = load_target()
    st.markdown(
        "Truth series from `v1/target.parquet`: mFRR activated **energy** (net, MWh per "
        "quarter-hour, upward positive) and mFRR **price** (EUR/MWh), Spain, 15-min, "
        "10 Dec 2024 – 31 Oct 2026. Missing energy is filled with 0 (no activation); missing "
        "price stays empty. Times in Madrid local time.\n\n"
        "- **Iberian blackout:** no data 28 Apr – 6 May 2025 (energy 0, price missing).\n"
        "- **Regime shift:** prices rise from Jul 2026 (monthly mean ~100 → ~157 EUR/MWh through "
        "Oct 2026), so recent months look different from the rest of the history."
    )
    target = st.radio("Target", ["energy", "price"], horizontal=True, key="tgt")
    unit = "MWh" if target == "energy" else "EUR/MWh"
    v = tv[tv.target == target]

    st.subheader("Time series")
    c1, c2 = st.columns([1, 2])
    res = c1.radio("Resolution", ["Daily", "15-min"], horizontal=True)
    if res == "Daily":
        d = v.groupby("day").value.agg(["mean", "min", "max"]).reset_index()
        base = alt.Chart(d).encode(x=alt.X("day:T", title=None))
        chart = base.mark_area(opacity=0.2).encode(
            y=alt.Y("min:Q", title=unit), y2="max:Q"
        ) + base.mark_line().encode(
            y="mean:Q",
            tooltip=[alt.Tooltip("day:T"), alt.Tooltip("mean:Q", format=".1f"), "min", "max"],
        )
        st.caption("Daily mean (line) and min–max (band).")
    else:
        lo, hi = v.day.min().date(), v.day.max().date()
        rng = c2.date_input(
            "Days (max 31)", value=(hi - dt.timedelta(days=13), hi), min_value=lo, max_value=hi
        )
        a, b = (rng[0], rng[-1]) if rng else (hi, hi)
        b = min(b, a + dt.timedelta(days=30))
        w = v[(v.day >= pd.Timestamp(a)) & (v.day <= pd.Timestamp(b))]
        chart = (
            alt.Chart(w)
            .mark_line()
            .encode(
                x=alt.X("time:T", title=None),
                y=alt.Y("value:Q", title=unit),
                tooltip=["time:T", alt.Tooltip("value:Q", format=".1f")],
            )
        )
    st.altair_chart(chart.interactive(bind_y=False))

    st.subheader("Typical patterns")
    st.caption("Mean (line) and q10–q90 (band) over the whole history.")
    cols = st.columns(3)
    cols[0].altair_chart(
        band_chart(
            profile(target, "slot"),
            alt.X(
                "slot:O", title=None, axis=alt.Axis(values=[f"{h:02d}:00" for h in range(0, 24, 3)])
            ),
            "Quarter-hour of day",
        )
    )
    wd = profile(target, "weekday").assign(
        weekday=lambda d: d.weekday.map(dict(enumerate(WEEKDAYS)))
    )
    cols[1].altair_chart(band_chart(wd, alt.X("weekday:O", title=None, sort=WEEKDAYS), "Weekday"))
    cols[2].altair_chart(
        band_chart(profile(target, "month"), alt.X("yearmonth(month):T", title=None), "Month")
    )

    st.subheader("Distribution")
    x = v.value.dropna()
    stats = {
        "Mean": f"{x.mean():.1f}",
        "Std": f"{x.std():.1f}",
        "Exact zeros": f"{(x == 0).mean():.1%}",
        "Negative": f"{(x < 0).mean():.1%}",
        "Min / max": f"{x.min():.0f} / {x.max():.0f}",
    }
    for col, (k, s) in zip(st.columns(len(stats)), stats.items(), strict=True):
        col.metric(k, s)
    lo, hi = x.quantile([0.005, 0.995])
    counts, edges = np.histogram(x.clip(lo, hi), bins=80)
    hist = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})
    st.altair_chart(
        alt.Chart(hist)
        .mark_bar()
        .encode(
            x=alt.X("start:Q", bin="binned", title=unit),
            x2="end:Q",
            y=alt.Y("count:Q", title="Quarter-hours"),
            tooltip=[
                alt.Tooltip("start:Q", format=".1f"),
                alt.Tooltip("end:Q", format=".1f"),
                "count",
            ],
        )
    )
    st.caption("Values clipped to the 0.5–99.5% range for display (extremes pile up at the edges).")

    st.subheader("Missing values")
    kind = st.radio("Count per day", ["Missing", "Exact zeros"], horizontal=True)
    flag = v.value.isna() if kind == "Missing" else v.value.eq(0)
    per_day = flag.groupby(v.day).sum().rename("n").reset_index()
    per_day["month"] = per_day.day.dt.strftime("%Y-%m")
    per_day["dom"] = per_day.day.dt.day
    st.altair_chart(
        alt.Chart(per_day)
        .mark_rect()
        .encode(
            x=alt.X("dom:O", title="Day of month"),
            y=alt.Y("month:O", title=None),
            color=alt.Color(
                "n:Q",
                scale=alt.Scale(scheme="oranges", domain=[0, max(1, per_day.n.max())]),
                title="Quarter-hours",
            ),
            tooltip=[alt.Tooltip("day:T"), "n"],
        )
    )
    st.caption(
        f"{int(flag.sum())} of {len(flag)} quarter-hours ({flag.mean():.1%}). Energy gaps were "
        "filled with 0 upstream, so for energy use 'Exact zeros' to see them."
    )

    st.subheader("Price vs energy")
    both = pd.DataFrame(
        {t: tv[tv.target == t].value.to_numpy() for t in ["energy", "price"]}
    ).dropna()
    sample = both.sample(min(5000, len(both)), random_state=0)
    st.altair_chart(
        alt.Chart(sample)
        .mark_circle(size=8, opacity=0.3)
        .encode(
            x=alt.X("energy:Q", title="Net mFRR energy (MWh)"),
            y=alt.Y("price:Q", title="mFRR price (EUR/MWh)"),
            tooltip=["energy", "price"],
        )
    )
    st.caption("Random sample of 5000 quarter-hours.")
