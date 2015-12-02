help:
	@echo "usage: make <tag>"
	@echo
	@echo "setup - get environment ready to run"
	@echo "test  - run all of the tests"
	@echo "check - check code for style"
	@echo "todo  - list all XXX, TODO and FIXME flags"
	@echo "ubuntu-setup - install necessary packages on ubuntu and setup (requires root)"

test:
	. bin/activate && nosetests test

check:
	./bin/flake8 gameanalysis

todo:
	grep -nrIF -e TODO -e XXX -e FIXME * --exclude-dir=lib --exclude-from=Makefile --color=always

setup:
	virtualenv -p python3 .
	bin/pip3 install -U pip
	bin/pip3 install -r requirements.txt

ubuntu-requirements:
	sudo apt-get install python3 libatlas-base-dev gfortran
	sudo pip3 install virtualenv

ubuntu-setup: ubuntu-requirements setup

.PHONY: test
