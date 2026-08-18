.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python main.py

debug:
	uv run python -m pdb main.py

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