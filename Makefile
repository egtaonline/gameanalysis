PYTEST_ARGS = -nauto --strict --showlocals -c/dev/null
FILES = gameanalysis/[a-z]* test setup.py

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup    - setup environment for developing"
	@echo "test     - run the tests, add file=<file> to run on a specific file e.g. file=rsgame"
	@echo "big      - run all of the tests (may fail on some computers)"
	@echo "coverage - run the tests and print coverage, add file=<file> to run on specific file"
	@echo "check    - check code for style"
	@echo "format   - try to autoformat code"
	@echo "todo     - list all XXX, TODO and FIXME flags"
	@echo "docs     - generate html for documentation"
	@echo "serve    - Serve documentation"
	@echo "ubuntu-reqs - install necessary packages on ubuntu (requires root)"

test:
ifdef file
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS) 2>/dev/null
else
	bin/py.test test $(PYTEST_ARGS) 2>/dev/null
endif

coverage:
ifeq ($(file),scripts)
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS) --cov gameanalysis.script --cov gameanalysis.scriptutils --cov gameanalysis/scripts --cov test.$(file)_test --cov-report term-missing 2>/dev/null
else
ifdef file
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS) --cov gameanalysis.$(file) --cov test.$(file)_test --cov-report term-missing 2>/dev/null
else
	bin/py.test test $(PYTEST_ARGS) --cov gameanalysis --cov test --cov-report term-missing 2>/dev/null
endif
endif

big: export BIG_TESTS=ON
big: test

check:
	bin/flake8 $(FILES)

format:
	bin/autopep8 -ri $(FILES)

todo:
	grep -nrIF -e TODO -e XXX -e FIXME --color=always README.md $(FILES)

setup:
	pyvenv .
	bin/pip install -U pip setuptools
	bin/pip install -e .
	bin/pip install -r requirements.txt

ubuntu-reqs:
	sudo apt-get install python3 libatlas-base-dev gfortran python3-venv moreutils jq

docs:
	$(MAKE) -C docs html

serve: docs
	cd docs/build/html && ../../../bin/python -m http.server

upload:
	cp ~/.pypirc ~/.pypirc.bak~ || touch ~/.pypirc.bak~
	echo '[distutils]\nindex-servers =\n    pypi\n\n[pypi]\nrepository: https://pypi.python.org/pypi\nusername: strategic.reasoning.group' > ~/.pypirc
	bin/python setup.py sdist bdist_wheel upload; mv ~/.pypirc.bak~ ~/.pypirc

clean:
	rm -rf bin include lib lib64 share pyvenv.cfg

.PHONY: test big docs
