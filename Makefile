# futmarket — common commands.
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

.PHONY: help install dashboard serve collect-bulk picks scorecard train \
        build-dataset test clean autonomous-install autonomous-status autonomous-log

help: ## Show this help
	@echo "futmarket — make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Using interpreter: $(PY)"

install: ## Install the project (+ web/ml extras) into the venv
	uv sync --extra web --extra ml --extra dev 2>/dev/null || $(PY) -m pip install -e '.[web,ml,dev]'

dashboard: ## Launch the live dashboard at HOST:PORT (open /ml)
	$(FUT) dashboard --host $(HOST) --port $(PORT)

serve: dashboard ## Alias for `dashboard`

collect-bulk: ## Snapshot the whole market's prices in one pass
	$(FUT) collect-bulk

picks: ## What to buy right now, at what price, and why
	$(FUT) picks

scorecard: ## Score past picks and show the real track record
	$(FUT) scorecard

train: ## Train + walk-forward validate the models
	$(FUT) train

build-dataset: ## Assemble the ML feature matrix
	$(FUT) build-dataset

test: ## Run the test suite
	$(PY) -m pytest -q

clean: ## Remove caches (keeps data/ and the DB)
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__

autonomous-install: ## Install the 24/7 ML loop as a macOS LaunchAgent
	bash scripts/install_ml.sh

autonomous-status: ## Show whether the ML/dashboard/awake agents are loaded
	@launchctl list | grep futmarket || echo "no futmarket agents loaded"
	@echo "--- recent ML cycle log ---"; tail -n 4 data/ml_daily.log 2>/dev/null || echo "(no log yet)"

autonomous-log: ## Follow the ML cycle log
	tail -f data/ml_daily.log
