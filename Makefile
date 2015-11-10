help:
	@echo "usage: make <tag>"
	@echo
	@echo "test : run all of the tests"
	@echo "ubuntu-setup : setup a clean installation on ubuntu (requires root)"

test:
	. bin/activate && nosetests test

ubuntu-setup:
	sudo apt-get install python3 libatlas-base-dev
	sudo pip3 install virtualenv
	virtualenv -p python3 .
	. bin/activate && pip3 install -r requirements.txt

.PHONY: test
