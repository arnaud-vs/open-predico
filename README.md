# open-predico

Open-source forecasting for the [Predico](https://predico-elia.inesctec.pt/docs/) collaborative forecasting platform, and an open benchmark built from it.

- **First targets:** Spanish mFRR balancing energy and price (day-ahead, 15-min).
- **Engines:** time-series foundation models, benchmarked against ML models and simple baselines.
- **Open:** data pipelines, features, models and results are public. Anyone can contribute.

Status: early. See [PLAN.md](PLAN.md) for the design and build order.

## Quickstart

```bash
git clone https://github.com/arnaud-vs/open-predico && cd open-predico
uv sync
uv run pytest
```

## Contributing

Any improvement is welcome — data, features, models, docs, tests, or the repo structure itself. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

Independent project. Not affiliated with INESC TEC, Elia, 50Hertz or EDP.

## License

Code: [Apache-2.0](LICENSE). Data sources keep their own licenses, see [DATA_SOURCES.md](DATA_SOURCES.md).
