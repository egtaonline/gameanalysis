PYLINT_ARGS =
PYTEST_ARGS =
PYTHON = python3
PYTHON_DIRS = gameanalysis test profile

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup   - setup environment for developing"
	@echo "test    - run the tests and print coverage"
	@echo "check   - check code for style"
	@echo "nash    - compute nash method profiling"
	@echo "docs    - generate html for documentation"
	@echo "publish - upload package to pypi"
	@echo "clean   - remove build objects"
	@echo "ubuntu-reqs - install necessary packages on ubuntu (requires root)"

test:
	bin/pytest test $(PYTEST_ARGS) --cov gameanalysis --cov test 2>/dev/null

format:
	bin/black $(PYTHON_DIRS)

check:
	bin/black --check $(PYTHON_DIRS)
	# pylint is no longer compliant
	bin/pylint $(PYLINT_ARGS) $(PYTHON_DIRS) || true

nash:
	bin/python profile/run.py 20 | tee profile/data.json | bin/python profile/display.py > sphinx/profile_nash.rst

setup:
	$(PYTHON) -m venv .
	bin/pip install -U pip setuptools
	bin/pip install -e '.[dev,nn]'

ubuntu-reqs:
	sudo apt-get install python3 libatlas-base-dev gfortran python3-venv moreutils jq

docs:
	rm -f sphinx/gameanalysis.rst sphinx/gameanalysis.*.rst
	bin/python setup.py build_sphinx -b html

publish:
	rm -rf dist
	bin/python setup.py sdist bdist_wheel
	bin/twine upload -u strategic.reasoning.group dist/*

clean:
	rm -rf bin build dist include lib lib64 share pyvenv.cfg gameanalysis.egg-info pip-selfcheck.json __pycache__ site-packages

travis: PYTEST_ARGS += -n2
travis: PYLINT_ARGS += -d fixme -j2
travis: check test

.PHONY: test check format todo nash setup ubuntu-reqs docs publish clean
