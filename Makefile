help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup    - get environment ready to run"
	@echo "test     - runs the quick tests"
	@echo "test-big - runs all of the tests (may fail on some computers)"
	@echo "check    - check code for style"
	@echo "todo     - list all XXX, TODO and FIXME flags"
	@echo "ubuntu-setup - install necessary packages on ubuntu and setup (requires root)"

test:
	bin/nosetests --rednose test

test-big: export BIG_TESTS = ON
test-big: test

check:
	# Eventually remove these exclusions
	bin/flake8 gameanalysis test --exclude Bootstrap.py,Sequential.py

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude=Makefile --color=always

setup:
	virtualenv -p python3.5 .
	bin/pip3 install -U pip --ignore-installed
	bin/pip3 install -r requirements.txt

ubuntu-requirements:
	sudo apt-get install python3.5 libatlas-base-dev gfortran
	sudo pip3 install virtualenv

ubuntu-setup: ubuntu-requirements setup

.PHONY: test
