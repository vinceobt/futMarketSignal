# FUT Market Desk — common commands.
# Uses the project virtualenv directly, so you never need to `source .venv/bin/activate`.
# Run `make` or `make help` to see everything.

# Prefer the repo venv; fall back to uv, then the system interpreter.
PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
              elif command -v uv >/dev/null 2>&1; then echo "uv run python"; \
              else echo python3; fi)
FUT := $(PY) -m futmarket
HOST ?= 127.0.0.1
PORT ?= 8000

.DEFAULT_GOAL := help

.PHONY: help install dashboard serve collect backtest signals players build-features test clean

help: ## Show this help
	@echo "FUT Market Desk — make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Using interpreter: $(PY)"

install: ## Install the project (+ web extras) into the venv
	uv sync --extra web --extra dev 2>/dev/null || $(PY) -m pip install -e '.[web,dev]'

dashboard: ## Launch the web dashboard (API + scraper worker) at HOST:PORT
	$(FUT) dashboard --host $(HOST) --port $(PORT)

serve: dashboard ## Alias for `dashboard`

collect: ## Run one polite scrape pass over the whole watchlist
	$(FUT) collect-once

backtest: ## Backtest the signal rule vs baselines on stored history
	$(FUT) backtest

signals: ## Evaluate + store current BUY/SELL/HOLD signals
	$(FUT) signals

players: ## List tracked players with snapshot counts
	$(FUT) players

build-features: ## Recompute the features table from stored prices
	$(FUT) build-features

test: ## Run the test suite
	$(PY) -m pytest -q

clean: ## Remove caches (keeps data/ and the DB)
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__
