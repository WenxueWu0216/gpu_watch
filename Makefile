# GPUWatch developer entry points. Everything here runs on the simulator; no GPU needed.
PY ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
GO ?= go

.PHONY: setup test test-py test-go eval eval-check lint fmt build-exporter compose-up compose-down validate clean

setup: ## create venv and install all deps (incl. dev/otel/llm extras)
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev,otel,llm]"
	cd exporter && $(GO) mod download

test: test-py test-go ## run all unit tests

test-py:
	$(BIN)/pytest --cov=gpu_sim --cov=detector --cov=bringup --cov=tracing --cov=ai_diagnosis --cov=eval --cov-report=term-missing:skip-covered --cov-fail-under=85

test-go:
	cd exporter && $(GO) test -race -cover ./...

eval: ## run fault-injection eval, write docs/results.{md,json} + README table, enforce CI gate
	$(BIN)/python -m eval

eval-check: eval ## CI: eval must pass AND committed results must match regenerated ones
	git diff --exit-code -- docs/results.md docs/results.json README.md

lint: ## ruff + mypy + gofmt + go vet
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy
	@test -z "$$(gofmt -l exporter)" || (echo "gofmt needed:"; gofmt -l exporter; exit 1)
	cd exporter && $(GO) vet ./...

fmt:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .
	gofmt -w exporter

build-exporter:
	cd exporter && CGO_ENABLED=0 $(GO) build -o bin/gpuwatch-exporter .

validate: ## validate dashboards, alert rules, helm chart, terraform (needs promtool/helm/terraform on PATH)
	$(BIN)/pytest -q tests/test_dashboards.py
	promtool check rules dashboards/alerts/*.yaml
	promtool test rules dashboards/alerts/tests/*.yaml
	helm lint deploy/helm/gpuwatch
	cd deploy/terraform && terraform init -backend=false -input=false >/dev/null && terraform validate

compose-up: ## full local stack on the simulator
	docker compose up --build -d

compose-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov exporter/bin reports
