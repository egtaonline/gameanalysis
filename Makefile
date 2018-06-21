PYLINT_ARGS =
PYTEST_ARGS =
PYTHON = python3

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup   - setup environment for developing"
	@echo "test    - run the tests and print coverage"
	@echo "check   - check code for style"
	@echo "docs    - generate html for documentation"
	@echo "publish - upload package to pypi"
	@echo "clean   - remove build objects"
	@echo "ubuntu-reqs - install necessary packages on ubuntu (requires root)"

test:
	bin/pytest test $(PYTEST_ARGS) --cov gameanalysis --cov test 2>/dev/null

check:
	bin/pylint $(PYLINT_ARGS) gameanalysis test

setup:
	$(PYTHON) -m venv .
	bin/pip install -U pip setuptools
	bin/pip install -e '.[dev,nn]'

ubuntu-reqs:
	sudo apt-get install python3 libatlas-base-dev gfortran python3-venv moreutils jq

docs:
	bin/python setup.py build_sphinx -b html

publish:
	rm -rf dist
	bin/python setup.py sdist bdist_wheel
	bin/twine upload -u strategic.reasoning.group dist/*

clean:
	rm -rf bin build dist include lib lib64 share pyvenv.cfg gameanalysis.egg-info pip-selfcheck.json __pycache__ site-packages

travis: PYTEST_ARGS += -v -n2
travis: PYLINT_ARGS += -d fixme -j2
travis: check test

.PHONY: test check format todo setup ubuntu-reqs docs publish clean
