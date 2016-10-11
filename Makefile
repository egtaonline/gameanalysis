PYTEST_ARGS = -n auto --strict --showlocals

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
	@echo "minor    - commit a minor version"
	@echo "major    - commit a major version"
	@echo "ubuntu-reqs - install necessary packages on ubuntu (requires root)"

test:
ifdef file
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS)
else
	bin/py.test test $(PYTEST_ARGS)
endif

coverage:
ifeq ($(file),scripts)
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS) --cov gameanalysis/$(file) --cov test/$(file)_test.py --cov-report term-missing
else
ifdef file
	bin/py.test test/$(file)_test.py $(PYTEST_ARGS) --cov gameanalysis/$(file).py --cov test/$(file)_test.py --cov-report term-missing
else
	bin/py.test test $(PYTEST_ARGS) --cov gameanalysis --cov test --cov-report term-missing
endif
endif

big: export BIG_TESTS=ON
big: test

check:
	bin/flake8 gameanalysis test

format:
	bin/autopep8 -ri gameanalysis test

todo:
	grep -nrIF -e TODO -e XXX -e FIXME . --exclude-dir=.git --exclude-dir=lib --exclude=Makefile --color=always

setup:
	pyvenv .
	bin/pip install -U pip setuptools
	bin/pip install -e .
	bin/pip install -r requirements.txt

ubuntu-reqs:
	sudo apt-get install python3 libatlas-base-dev gfortran python3-venv moreutils jq

bump-minor:
	jq '.version = (.version | split(".") | .[1] = (.[1] | tonumber + 1 | tostring) | join("."))' setup.json | sponge setup.json

bump-major:
	jq '.version = (.version | split(".") | [.[0] | tonumber + 1 | tostring, "0"] | join("."))' setup.json | sponge setup.json

bump-sync:
	sed -ri.un~ "s/^version = '[0-9]+\.[0-9]+'$$/version = '$(shell jq -r '.version' setup.json)'/;s/^release = '[0-9]+\.[0-9]+'$$/release = '$(shell jq -r '.version' setup.json)'/" docs/source/conf.py
	bin/sphinx-apidoc -f -o docs/source gameanalysis
	$(MAKE) -C docs html
	cd docs/build/html && git add . && git commit -m 'Update pages to $(shell jq -r .version setup.json)'; git push origin gh-pages
	git commit setup.json docs/source/conf.py docs/build/html
	git tag v$(shell jq -r .version setup.json)
	git push $(shell git remote | head -n1) v$(shell jq -r .version setup.json)

minor: bump-minor bump-sync

major: bump-major bump-sync

clean:
	rm -rf bin include lib lib64 share

.PHONY: test big
