Game Analysis
=============

[![Build Status](https://travis-ci.org/egtaonline/gameanalysis.svg?branch=master)](https://travis-ci.org/egtaonline/gameanalysis)
[![Coverage Status](https://coveralls.io/repos/github/egtaonline/gameanalysis/badge.svg?branch=master)](https://coveralls.io/github/egtaonline/gameanalysis?branch=master)
[![Documentation Status](https://readthedocs.org/projects/gameanalysis/badge/?version=latest)](http://gameanalysis.readthedocs.io/en/latest/?badge=latest)


This is a collection of python libraries and scripts that manipulate empirical game data.


Quick Setup
-----------

We recommend you install Game Analysis in it's own virtual environment.
To use our recommended setup simply execute the following commands in the directory you want to store Game Analysis in.

```
curl https://raw.githubusercontent.com/egtaonline/gameanalysis/master/quickuse_makefile > Makefile && make setup
```

`ga` should now be accessible in the `bin` directory.
To update Game Analysis, simply execute `make update` in the appropriate directory.


Setup
-----

To use Game Analysis, you need to meet the following dependencies

1. Python 3 & venv
2. BLAS/LAPACK
3. A fortran compiler


Then you can install Game Analysis via pip with:

```
bin/pip install -U git+https://github.com/egtaonline/gameanalysis.git@<version>
```

where `<version>` is the appropriate version to install.
Generally we recommend this be done in a virtual environment to avoid dependency clashes, but it can be installed in the global environment.


Usage
-----

`ga` is the game analysis command line tool.
`./ga --help` will reveal all of the available options.


Developing
==========

After cloning this repository, the included `Makefile` includes all the relevant actions to facilitate development.
Typing `make` without targets will print out the various actions to help development.


Testing
-------

All of the tests can be run with `make test`.
If you want more fine grained control, you can run `make test file=<file>` to execute tests for a single file in game analysis e.g. `make test file=rsgame`.
Additionally, `make coverage` and `make coverage file=<file>` will run all of the tests and output a report on the coverage.


Games
-----

There are three game types: BaseGame, Game, and SampleGame.

BaseGame contains several functions that are valid for games without payoff data, and has the general structure that arbitrary game-like objects should inherit from.

Game is a potentially sparse mapping from role symmetric profiles to payoffs.
It provides methods to quickly calculate mixture deviation gains, necessary for computing nash equilibria.

SampleGame retains payoff data for every observation.
This allows it to resample the payoff data for every individual profile.


Profiles
--------

Internally this library uses arrays to store game profiles, and doesn't care about the names attached to a role or strategy, only their index. For consistence of lexicographic tie-breaking, roles and strategies are indexed in lexicographic order when serializing a named game into an internal array representation.


Style Guidelines
----------------

Generally follow PEP8 standard.

1. Single quotes
2. Lowercase underscore for mathod names
3. Camelcase classes
4. Unless obvious or necessary, try to only import modules not specific
   functions or classes from a module.
5. Put a docstring for every public function and class. The first line should
   be short summary followed by a more detailed description perhaps detailing
   information about parameters or return values.
6. flake8

Running `make check` will search for some of these.
`make format` will try to fix some in place.


To Do
-----

- Change conditional in `dominance`, which indicates how to treat missing data to an enum or at least a string
- Some functions in `dominance` could probably be more efficient.
- Using array set operations would allow for convenient array operations like, "are all of these profiles present", however, it requires sorting of large void types which is very expensive, less so than just hashing the data. Maybe with pandas?
- Test requirements are also in requirements.txt because of issues loading them with xdist.
