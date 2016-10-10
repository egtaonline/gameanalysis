Game Analysis
=============

This is a set of python scripts to manipulate empirical game data.


Setup
-----

To use this script, you need to install the following dependencies:

1. Python 3
2. BLAS/LAPACK
3. A fortran compiler
4. Make

### Ubuntu

These dependencies can be met with

```
$ sudo apt install python3 libatlas-base-dev gfortran python3-venv
```

or `make ubuntu-reqs`

### Mac

On mac you can install these easily with [homebrew](http://brew.sh/).

TODO add actual setup commands

### Final setup

After all of the dependencies are met, executing

```
$ make update
```

will complete the setup.


Usage
-----

`ga` is the game analysis command line tool.
`./ga --help` will reveal all of the available options.
If the root of this project is on your python path (done manually, with the venv active, or when executing anything in `bin`), then you also import individual packages from `gameanalysis`.


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
- Integrate read the docs with numpy docstyle extension and github travis-ci for testing etc.
- Using array set operations would allow for convenient array operations like, "are all of these profiles present", however, it requires sorting of large void types which is very expensive, less so than just hashing the data. Maybe with pandas?
