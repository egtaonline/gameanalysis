PYTEST_ARGS = -n auto --strict

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup    - get environment ready to run"
	@echo "test     - run the tests, add file=<file> to run on a specific file e.g. file=rsgame"
	@echo "big      - run all of the tests (may fail on some computers)"
	@echo "coverage - run the tests and print coverage, add file=<file> to run on specific file"
	@echo "check    - check code for style"
	@echo "todo     - list all XXX, TODO and FIXME flags"
	@echo "ubuntu-setup - install necessary packages on ubuntu and setup (requires root)"

test:
ifdef file
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS)
else
	bin/py.test test $(PYTEST_ARGS)
	test/command_line_test.sh
endif

coverage:
ifdef file
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS) --cov gameanalysis.$(file) --cov test.$(file)_test --cov-report term-missing
else
	bin/py.test test $(PYTEST_ARGS) --cov gameanalysis --cov test --cov-report term-missing
endif

big: export BIG_TESTS=ON
big: test

check:
	bin/flake8 gameanalysis test

format:
	bin/autopep8 -ri gameanalysis test

todo:
	grep -nrIF -e TODO -e XXX -e FIXME . --exclude-dir=lib --exclude=Makefile --color=always

setup:
	git pull
	git submodule update --init --recursive
	virtualenv -p python3 .
	bin/pip3 install -UI pip
	bin/pip3 install -r requirements.txt

ubuntu-requirements:
	sudo apt-get install python3 libatlas-base-dev gfortran
	sudo pip3 install virtualenv

ubuntu-setup: ubuntu-requirements setup

.PHONY: test big
