FILES = gameanalysis test setup.py
PYTHON = python3

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup   - setup environment for developing"
	@echo "test    - run the tests and print coverage"
	@echo "check   - check code for style"
	@echo "format  - try to autoformat code"
	@echo "todo    - list all XXX, TODO and FIXME flags"
	@echo "docs    - generate html for documentation"
	@echo "publish - upload package to pypi"
	@echo "clean   - remove build objects"
	@echo "ubuntu-reqs - install necessary packages on ubuntu (requires root)"

test:
	bin/pytest test --cov gameanalysis --cov test 2>/dev/null

check:
	bin/flake8 $(FILES)

format:
	bin/autopep8 -ri $(FILES)

todo:
	grep -nrIF -e TODO -e XXX -e FIXME --color=always README.md $(FILES)

setup:
	$(PYTHON) -m venv .
	bin/pip install -U pip setuptools
	bin/pip install -e '.[dev,nn]'

ubuntu-reqs:
	sudo apt-get install python3 libatlas-base-dev gfortran python3-venv moreutils jq

docs:
	bin/python setup.py build_sphinx -b html

publish:
	bin/python setup.py sdist bdist_wheel
	bin/twine upload -u strategic.reasoning.group dist/*

clean:
	rm -rf bin build dist include lib lib64 share pyvenv.cfg gameanalysis.egg-info pip-selfcheck.json __pycache__ site-packages

.PHONY: test check format todo setup ubuntu-reqs docs publish clean
