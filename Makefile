TEST_WORKERS=4
CODE=src
TESTS=tests
COVERAGE_PERCENT=0

ALL = $(CODE) $(TESTS)

pretty:
	isort $(ALL)
	black $(ALL)
	toml-sort pyproject.toml

lint:
	black --check $(ALL)
	mypy $(ALL)
	toml-sort --check pyproject.toml

test:
	pytest -n 4 $(TESTS) --random-order-bucket=module

coverage:
	pytest $(TESTS) --cov $(CODE) --cov-fail-under $(COVERAGE_PERCENT) --cov-report html
