GOINFRE := /goinfre/$(USER)

export UV_CACHE_DIR := $(GOINFRE)/.cache/uv
export UV_PROJECT_ENVIRONMENT := $(GOINFRE)/.venvs/callmemaybe
export HF_HOME := $(GOINFRE)/.cache/huggingface
export MYPY_CACHE_DIR := $(GOINFRE)/.cache/mypy
export TMPDIR := $(GOINFRE)/tmp
unexport VIRTUAL_ENV
$(shell mkdir -p $(TMPDIR))

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 . --exclude=.venv,llm_sdk
	uv run mypy . --exclude '(^|/)(\.venv|llm_sdk)/' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=.venv,llm_sdk
	uv run mypy . --strict --exclude '(^|/)(\.venv|llm_sdk)/'