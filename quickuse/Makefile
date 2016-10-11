update:
	bin/pip install -U git+https://github.com/egtaonline/GameAnalysis.git@$(shell curl https://api.github.com/repos/egtaonline/GameAnalysis/tags 2>/dev/null | jq -r 'map(.name | select(startswith("v")) | .[1:] | split(".") | map(tonumber)) | sort | .[-1] | map(tostring) | "v" + join(".")')

setup: setup-pre update

setup-pre:
ifeq ($(shell uname),Darwin)
	@echo mac setup not finished
else #Linux
	which pyvenv 2>&1 >/dev/null || sudo apt-get install python3-venv
	which jq 2>&1 >/dev/null || sudo apt-get install jq
	which gfortran 2>&1 >/dev/null || sudo apt-get install gfortran
	{ ldconfig -p | grep liblapack 2>&1 >/dev/null && ldconfig -p | grep libblas 2>&1 >/dev/null; } || sudo apt-get install libatlas-base-dev
endif
	pyvenv .
	bin/pip install -U pip setuptools
