PYTHON ?= python

.PHONY: test
test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s tests -q
