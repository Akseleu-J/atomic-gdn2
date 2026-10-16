.PHONY: install dev test test-fast lint fmt print-config
install:        ## CPU/dev install
	pip install -r requirements.txt && pip install -e ".[dev]"
test-fast:      ## no Pallas kernels (seconds)
	pytest -m "not kernel"
test:           ## everything incl. Pallas interpret-mode tests (minutes on CPU)
	pytest
lint:
	ruff check src tests scripts
fmt:
	ruff check --fix src tests scripts
print-config:
	python -m atomic_gdn2.cli --config configs/hybrid_hier_1024_28l.yaml --print-config
