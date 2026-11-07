"""Read-only live check; prints metadata only and never uploads data or submits forecasts."""

import json

import pandas as pd

from openpredico.data.fiftyhertz import fetch_frame


def main():
    # Yesterday is complete in local calendar time; its latest publication can still lag.
    end = pd.Timestamp.now(tz="Europe/Berlin").normalize()
    start = end - pd.DateOffset(days=1)
    frame = fetch_frame("actual", start, end)
    expected = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    if not frame.index.equals(expected):
        raise RuntimeError("Latest complete day's timestamp coverage is incomplete")
    if frame.isna().any().any():
        raise RuntimeError("Missing actual wind values; no filling is performed")
    print(
        json.dumps(
            {
                "source": "50Hertz",
                "local_day": start.strftime("%Y-%m-%d"),
                "intervals": len(frame),
                "timezone": str(frame.index.tz),
                "columns": list(frame.columns),
                "missing_values": int(frame.isna().sum().sum()),
                "status": "PASS",
            }
        )
    )


if __name__ == "__main__":
    main()
