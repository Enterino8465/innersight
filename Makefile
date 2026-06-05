# InnerSight — developer & demo Makefile.
# Run `make help` to see the available targets.

VENV    := $(CURDIR)/backend/.venv/bin
PYTHON  := $(if $(wildcard $(VENV)/python),$(VENV)/python,python)
RUFF    := $(if $(wildcard $(VENV)/ruff),$(VENV)/ruff,ruff)
COMPOSE := $(shell if command -v docker-compose >/dev/null 2>&1; then echo docker-compose; else echo docker compose; fi)

.DEFAULT_GOAL := help
.PHONY: demo stop test lint train-quick clean help

help: ## Show this help.
	@echo "InnerSight — available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

demo: ## Build & start the full stack (Docker); open http://localhost:3000
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker is not installed / not on PATH."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon is not running. Start Docker Desktop and retry."; exit 1; }
	@for p in 3000 5001 6333; do \
		if lsof -iTCP:$$p -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "ERROR: port $$p is already in use. Free it (or run 'make stop') and retry."; exit 1; \
		fi; \
	done
	$(COMPOSE) up --build -d
	@echo "Waiting for the backend to become healthy (timeout 60s) ..."
	@t=60; while [ $$t -gt 0 ]; do \
		if curl -fsS http://localhost:5001/health >/dev/null 2>&1; then \
			echo ""; echo "InnerSight is running at http://localhost:3000"; exit 0; \
		fi; \
		sleep 2; t=$$((t-2)); \
	done; \
	echo "ERROR: backend did not become healthy within 60s. Check '$(COMPOSE) logs backend'."; exit 1

stop: ## Stop the stack.
	$(COMPOSE) down

test: ## Run the backend test suite.
	cd backend && $(PYTHON) -m pytest tests/ -x --tb=short

lint: ## Lint the backend with ruff.
	cd backend && $(RUFF) check innersight/ tests/

train-quick: ## Fast smoke training on the bundled demo data.
	cd backend && bash innersight/scripts/run_all_training.sh \
		--data-dir ../data/synthetic_demo --version r4.2 --quick

clean: ## Stop the stack and remove caches, checkpoints, and feature store.
	$(COMPOSE) down -v
	rm -rf backend/feature_store/
	rm -rf backend/checkpoints/*.pt
