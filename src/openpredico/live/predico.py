"""Minimal Predico API client. Docs: https://predico.inesctec.pt/docs/

Credentials from PREDICO_EMAIL / PREDICO_PASSWORD. Set PREDICO_URL to
https://playground.predico.inesctec.pt for dry runs.
Never log forecast values: GitHub Actions logs are public.
"""

import os

import pandas as pd
import requests

QUANTILES = ("q10", "q50", "q90")


class Predico:
    def __init__(self) -> None:
        self.api = (os.environ.get("PREDICO_URL") or "https://predico.inesctec.pt") + "/api/v1"
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
