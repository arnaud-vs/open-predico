"""Minimal Predico API client. Docs: https://predico.inesctec.pt/docs/

Credentials from PREDICO_EMAIL / PREDICO_PASSWORD; base URL from the env var named in URL_VAR.
Never log forecast values: live forecasts stay private until the gate closes.
"""

import os

import pandas as pd
import requests

QUANTILES = ("q10", "q50", "q90")
# production since 2026-11-13; "PREDICO_PLAYGROUND_URL" for dry runs
URL_VAR = "PREDICO_URL"


class Predico:
    def __init__(self) -> None:
        url = os.environ[URL_VAR]  # no silent fallback to production
        self.api = ("" if "://" in url else "https://") + url.rstrip("/") + "/api/v1"
        resp = requests.post(
            f"{self.api}/token",
            data={"email": os.environ["PREDICO_EMAIL"], "password": os.environ["PREDICO_PASSWORD"]},
            timeout=30,
        )
        resp.raise_for_status()
        # ponytail: access token lasts ~1 h, one run takes seconds; add refresh if runs get long
        self.headers = {
            "Authorization": f"Bearer {resp.json()['access']}",
            "Accept": "application/json",
        }

    def _get(self, path: str, **params) -> dict:
        resp = requests.get(f"{self.api}/{path}", params=params, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def resource_id(self, name: str) -> str:
        return self._get("user/resource/", resource_name=name)["data"][0]["id"]

    def submit_continuous(self, resource_id: str, fc: pd.DataFrame) -> None:
        """Send q10/q50/q90 columns of `fc` (UTC 15-min index), one request per quantile."""
        stamps = fc.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        for q in QUANTILES:
            body = {
                "variable": q,
                "forecasts": [
                    {"datetime": t, "value": round(float(v), 3)}
                    for t, v in zip(stamps, fc[q], strict=True)
                ],
            }
            resp = requests.post(
                f"{self.api}/market/continuous-forecasts/{resource_id}",
                json=body,
                headers=self.headers,
                timeout=30,
            )
            if resp.status_code != 201:
                # status + server message only; the body we sent contains forecast values
                raise RuntimeError(f"{q}: HTTP {resp.status_code}: {resp.text[:500]}")

    def coverage(self) -> dict:
        """Per-challenge readiness before the gate (`ready`, `source`)."""
        return self._get("market/session/coverage")

    def our_challenges(self, resource_name: str) -> list[dict]:
        """Challenges of the resource for which Predico registered a session submission of ours."""
        subs = {
            s["market_session_challenge"] for s in self._get("market/challenge/submission")["data"]
        }
        # ponytail: lists every challenge ever and callers fetch each; add a date cursor if slow
        challenges = self._get("market/challenge", resource_name=resource_name)["data"]
        return [c for c in challenges if c["id"] in subs]

    def submitted_forecasts(self, challenges: list[dict]) -> pd.DataFrame:
        """What Predico registered as our session submissions: q10/q50/q90 on a UTC index."""
        rows = []
        for c in challenges:
            rows += self._get("market/challenge/submission/forecasts", challenge=c["id"])["data"]
        if not rows:
            return pd.DataFrame(columns=list(QUANTILES), index=pd.DatetimeIndex([], tz="UTC"))
        df = pd.DataFrame(rows).assign(datetime=lambda d: pd.to_datetime(d.datetime, utc=True))
        df = df.sort_values("registered_at")  # a resubmission replaces the earlier one
        wide = df.pivot_table("value", "datetime", "variable", aggfunc="last")
        return wide.reindex(columns=list(QUANTILES)).rename_axis(index=None, columns=None)

    def scores(self, challenges: list[dict]) -> pd.DataFrame:
        """Predico's official scores of our submissions, with our rank among N participants.

        Only `personal_metrics`: other forecasters' results (`general_metrics`) are never kept.
        Challenges not evaluated yet return no rows.
        """
        rows = []
        for c in challenges:
            try:
                data = self._get("market/challenge/submission-scores", challenge=c["id"])["data"]
            except requests.HTTPError:
                continue  # not evaluated yet
            for m in data.get("personal_metrics", []) if isinstance(data, dict) else []:
                rows.append(
                    {
                        "day": str(c["target_day"])[:10],
                        "variable": m["variable"],
                        "metric": m["metric"],
                        "value": m["value"],
                        "rank": m["rank"],
                        "participants": m["total_participants"],
                    }
                )
        return pd.DataFrame(
            rows, columns=["day", "variable", "metric", "value", "rank", "participants"]
        )

    def open_challenges(self, resource_name: str) -> list[dict]:
        return self._get("market/challenge", open_only="true", resource_name=resource_name)["data"]
