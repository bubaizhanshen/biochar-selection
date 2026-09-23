PYTHON ?= python

.PHONY: test test-full
test:
	PYTHONPATH=code:$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -q

test-full:
	$(PYTHON) -c "import xgboost, lightgbm, catboost"
	$(MAKE) test
