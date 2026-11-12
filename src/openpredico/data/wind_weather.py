"""Separate point-weather source for the optional wind candidate; existing ECMWF is untouched.

Single-run responses remain separate. Callers archive raw records privately before inference.
The free endpoint is for research; commercial callers must supply their licensed endpoint.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import requests

MODELS = ("icon_eu", "icon_d2", "icon_global", "gfs_global")
BASE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"


def run_time(day: str, model: str) -> pd.Timestamp:
    if model not in MODELS:
        raise ValueError("unknown weather model")
    previous = pd.Timestamp(day) - pd.DateOffset(days=1)
    hour = 6 if model in ("icon_eu", "icon_d2") else 0
    return pd.Timestamp(f"{previous:%Y-%m-%d} {hour:02}:00", tz="UTC")


def fetch_run(
    day: str,
    model: str,
    points: list[tuple[float, float]],
    *,
    base_url: str = BASE_URL,
    api_key: str | None = None,
    commercial: bool = False,
) -> dict:
    """One bounded request for all points; return raw record, without logs or uploads.

    No automatic historical bootstrap or unbounded retries. A failure leaves the incumbent
    unchanged. Callers pace requests and validate retrieved_at <= their issue cutoff.
    """
    if commercial and base_url == BASE_URL:
        raise ValueError("commercial weather collection requires licensed access")
    if not points or len(points) > 12:
        raise ValueError("expected one to twelve weather points")
    run = run_time(day, model)
    fields = (
        ["wind_speed_80m", "wind_speed_120m", "wind_direction_120m"]
        if model.startswith("icon")
        else ["wind_speed_100m", "wind_direction_100m"]
    )
    if model == "icon_eu":
        fields += ["temperature_80m", "temperature_120m"]
    params = {
        "latitude": ",".join(str(p[0]) for p in points),
        "longitude": ",".join(str(p[1]) for p in points),
        "models": model,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "hourly": ",".join(fields),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    if api_key:
        params["apikey"] = api_key
    try:
        response = requests.get(base_url, params=params, timeout=75)
    except requests.RequestException:
        raise RuntimeError("point-weather request failed") from None
    if not response.ok:
        raise RuntimeError(f"point-weather HTTP {response.status_code}")  # never token-bearing URL
    locations = response.json()
    locations = locations if isinstance(locations, list) else [locations]
    if len(locations) != len(points):
        raise ValueError("point-weather location count differs")
    return {
        "model": model,
        "run_utc": run.isoformat(),
        "points": points,
        "responses": locations,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
    }


def decode_run(record: dict) -> pd.DataFrame:
    """Per-point speed100, sin/cos direction and EU T100, within-run 15-minute interpolation.

    Non-EU T100 is missing by design; the blend uses EU temperature for every component.
    Points and all raw response metadata remain in `record` for private archiving.
    """
    model = record["model"]
    if model not in MODELS:
        raise ValueError("unknown weather model")
    parts = []
    for n, response in enumerate(record["responses"], 1):
        frame = pd.DataFrame(response["hourly"])
        frame.index = pd.to_datetime(frame.pop("time"), utc=True)
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("invalid run timestamps")
        units = response["hourly_units"]
        field = "wind_speed_80m" if model.startswith("icon") else "wind_speed_100m"
        if units[field] != "m/s":
            raise ValueError("wind speed must be m/s")
        if model.startswith("icon"):
            a, b = frame.wind_speed_80m, frame.wind_speed_120m
            w = np.log(100 / 80) / np.log(120 / 80)
            speed = np.exp((1 - w) * np.log(a.clip(lower=1e-5)) + w * np.log(b.clip(lower=1e-5)))
            speed = speed.where(~((a == 0) & (b == 0)), 0)
            direction = frame.wind_direction_120m
        else:
            speed, direction = frame.wind_speed_100m, frame.wind_direction_100m
        temperature = np.nan
        if model == "icon_eu":
            if any(units[v] != "°C" for v in ("temperature_80m", "temperature_120m")):
                raise ValueError("temperature must be Celsius")
            temperature = (frame.temperature_80m + frame.temperature_120m) / 2
        parts.append(
            pd.DataFrame(
                {
                    f"speed_p{n}": speed,
                    f"sin_p{n}": np.sin(np.deg2rad(direction)),
                    f"cos_p{n}": np.cos(np.deg2rad(direction)),
                    f"t100_p{n}": temperature,
                }
            )
        )
    if not parts:
        raise ValueError("empty run")
    out = pd.concat(parts, axis=1)
    grid = pd.date_range(out.index.min(), out.index.max(), freq="15min")
    return out.reindex(grid).interpolate(method="time", limit=3, limit_area="inside")


def blend(raw: np.ndarray, speed_bank: np.ndarray, *, component: str) -> np.ndarray:
    """Fixed saved speed-band bank, retaining the tested per-component feature order.

    raw: (time, six points, MODELS, speed/sin/cos/T100). Temperature always comes from EU.
    Bank fitting and its pre-origin eligibility audit remain the caller's responsibility.
    """
    if component not in ("onshore", "offshore") or raw.shape[1:] != (6, 4, 4):
        raise ValueError("unexpected six-point weather layout")
    if speed_bank.shape != (6, 3, 4) or not np.isfinite(speed_bank).all():
        raise ValueError("invalid speed bank")
    if (speed_bank < 0).any() or not np.allclose(speed_bank.sum(axis=-1), 1):
        raise ValueError("weather weights must be convex")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmedian(raw[:, :, :, 0], axis=2)
    speed = np.where(np.isfinite(speed), speed, 0)
    membership = np.stack(
        [np.interp(speed, [2.5, 7.5, 12.5], np.eye(3)[b]) for b in range(3)], axis=-1
    )
    weights = np.einsum("tpb,pbm->tpm", membership, speed_bank)
    valid = np.isfinite(raw[:, :, :, :3]).all(axis=3)
    weights *= valid
    denominator = weights.sum(axis=2, keepdims=True)
    weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    wind = np.nansum(raw[:, :, :, :3] * weights[:, :, :, None], axis=2)
    fallback = (denominator[:, :, 0] <= 0) | ~np.isfinite(raw[:, :, 0, :3]).all(axis=2)
    wind[fallback] = raw[:, :, 0, :3][fallback]
    temperature = raw[:, :, 0, 3]
    if component == "onshore":
        return np.concatenate([wind, temperature[:, :, None]], axis=2).reshape(len(raw), 24)
    return np.concatenate([wind[:, :, k] for k in range(3)] + [temperature], axis=1)
