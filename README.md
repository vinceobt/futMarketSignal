# fc-market-analytics

Market intelligence for EA FC Ultimate Team player prices: tracks a personal
watchlist as time-series data and (in later phases) generates **buy/sell/hold
signals with human-readable rationale**, backtested before anyone trusts them.



## Setup

```bash
uv venv --python 3.12 .venv          # or python3.11+ -m venv .venv
uv pip install -e ".[dev]"           # or .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                     # 8 tests should pass
```

## Usage

```bash
futmarket players                    # watchlist + snapshot counts
futmarket log-price salah-base 87500 # record a price you saw (source=manual)
futmarket history salah-base         # snapshot table with Δ% between rows
futmarket collect-once               # one polite pass (needs a non-manual source)
futmarket run                        # polling loop, Ctrl-C to stop
```

## Configuration

Everything lives in `config.yaml` — watchlist (cap 50), platform
(`console`/`pc`), poll interval (floor 30 min), politeness knobs, circuit
breaker, paths. `player_id` slugs are yours to choose but must stay stable:
they key all stored history.

## Guarantees

- **Append-only history**: snapshots are never overwritten.
- **Idempotent collection**: a `UNIQUE(player_id, source, minute)` index plus
  a freshness skip-guard means re-runs and restarts can't duplicate rows.
- **Graceful degradation**: one player failing never stops the pass; a source
  failing repeatedly trips a breaker and cools down instead of hammering.
- **Structured logs**: every collect/skip/failure in `data/collector.log`.

## Build phases

- **Phase 0 (done)**: collector pipeline, storage, CLI, tests.
- **Phase 1**: feature engineering (momentum, z-score, market index
  normalization, event calendar distance, weekend-league flag).
- **Phase 2**: backtesting harness — no signal goes live without beating a
  naive baseline on historical data.
- **Phase 3**: rule-based signal engine with plain-language reasons, then
  Discord/Telegram alerts; ML only after the rules earn trust.

Each phase is gated on validating the previous one against real data.
