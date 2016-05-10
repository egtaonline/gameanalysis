NOSEFLAGS := --rednose

help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup    - get environment ready to run"
	@echo "test     - run the tests"
	@echo "test-big - run all of the tests (may fail on some computers)"
	@echo "coverage - run the tests and print coverage"
	@echo "check    - check code for style"
	@echo "todo     - list all XXX, TODO and FIXME flags"
	@echo "ubuntu-setup - install necessary packages on ubuntu and setup (requires root)"
	@echo "update-requirements - update requirements.txt with current pip packages"

test:
	bin/nosetests $(NOSEFLAGS) test$(tests)

test-big: export BIG_TESTS = ON
test-big: test

coverage: NOSEFLAGS += --with-coverage --cover-package gameanalysis
coverage: test

check:
	bin/flake8 gameanalysis test

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude=Makefile --color=always

setup:
	virtualenv -p python3 .
	bin/pip3 install -UI pip
	bin/pip3 install -r requirements.txt

ubuntu-requirements:
	sudo apt-get install python3 libatlas-base-dev gfortran
	sudo pip3 install virtualenv

ubuntu-setup: ubuntu-requirements setup

update-requirements:
	bin/pip3 freeze > requirements.txt

.PHONY: test
