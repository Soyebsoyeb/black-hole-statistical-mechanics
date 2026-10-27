.PHONY: install test test-fast experiments figures validate quickstart zip clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

test-fast:
	python -m pytest -q -m "not slow"

quickstart:
	python scripts/quickstart.py

experiments:
	python scripts/run_all_experiments.py --clean

figures: experiments

validate:
	python scripts/validate_results.py

zip:
	python scripts/package_zip.py

clean:
	rm -rf .pytest_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
