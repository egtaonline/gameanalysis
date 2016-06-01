Game Analysis
=============

This is a set of python scripts to manipulate empirical game data.


Setup
-----

You can follow the instructions below for how to setup your environment.
Alternatively, if you're using ubuntu you should just be able to execute `make
ubuntu-setup` from this directory to properly setup your environment. You will
need root privileges to execute this script.

Before this library can be used, you need to install several dependencies.

1. Python 3
2. BLAS/LAPACK
3. virtualenv

On ubuntu these dependencies can be installed with:

```
$ sudo apt-get install python3 libatlas-base-dev gfortran
$ sudo pip3 install virtualenv
```

On mac, similar programs should be able to be installed with [`brew`](brew.sh).
From here, setup virtualenv in this directory, and activate it with

```
make setup
```

At this point, you should run the tests to make sure everything was setup properly.
Executing `make test` should run all of the tests. If you see something like:

```
bin/nosetests --rednose test
.............................................................................

3221 tests run in 139.4 seconds (3221 tests passed)
```

Then all of the tests passed!
If you get failures with finding equilibria, it's probably fine.
Those tests can fail occasionally due to the random initialization.
Note, this may take a while to run.


Usage
-----

`ga` is the game analysis command line tool.
`./ga --help` will reveal all of the available options.
If the root of this project is on your python path (done manually, with the virtual env active, or when executing anything in `bin`), then you also import individual packages from `gameanalysis`.


Testing
-------

All of the tests can be run with `make test`.
If you want more fine grained control, you can run `bin/nosetests test.<unit-test-file-name>[:test-method-name]` to execute only a single test file, or or only a specific method from within a file.
You may also want to add the option `--nocapture` or `-s` to output `sysout` and `syserr`, which are usually captured.
Additionally, `make coverage` will run all of the tests and output a report on the coverage, which will look something like:

```
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
gameanalysis.py                 0      0   100%
gameanalysis/collect.py        73      0   100%
gameanalysis/gamegen.py       164      0   100%
gameanalysis/gameio.py         98     77    21%   12-26, 51-54, 59-60, 67-87, 92-101, 106-115, 120-131, 138-152
gameanalysis/nash.py           44      0   100%
gameanalysis/profile.py       100      0   100%
gameanalysis/reduction.py     128      0   100%
gameanalysis/regret.py         27      0   100%
gameanalysis/rsgame.py        506      0   100%
gameanalysis/subgame.py        78      0   100%
gameanalysis/utils.py         122      0   100%
---------------------------------------------------------
TOTAL                        1340     77    94%
```


Games
-----

There are three game types: EmptyGame, Game, and SampleGame.

EmptyGame contains several functions that are valid for games without payoff data.

Game is a potentially sparse mapping from role symmetric profiles to payoffs.
It behaves mostly like a python dictionary from profiles to payoff dictionaries.

SampleGame retains payoff data for every observation.
This allows it to resample the payoff data for every individual profile.


Profiles
--------

Profiles and Mixtures have a dictionary representation and a corresponding array representation that is only valid for the existing game object as it depends on the order that strategy names were hashed.
The existence of array versions is for efficiency of many internal operations.


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


Open Design Questions
---------------------

1. Whether to include version numbers or descriptions in json serialized profiles / mixtures or keep them a raw description.
2. How to handle games where data doesn't exist for every role strategy pair, but only some agents.
   Currently any incomplete profile is ignored / errors.
   There might not be an efficient way to handle this case.
3. Currently we use both array and dictionary representations of data, but it might make more sense to just use one.
   We could subclass ndarray, and modify the repr, str, items, keys, values so that they appear like a dictionary, but internally they are an array with a reference back to the original game.
   The pros are this simplifies a lot of things.
   We can just cast all input to this form, and always output this form, no more `as_array`.
   The downsides is that there are conflicts between the way an array operates and the way a dict operates, and it will be hard to join them in a way that makes sense. A couple that come to mind are:
   1. We need to handle ndarrays that have a number of profiles, but still handle appropriate methods.
   2. Equality on an array tests every value, versus the dict output.
      This could likely be acomplished by clever implementation of equality, i.e. equality with another profile or a dict returns a boolean, but equality with an array returns an array.
   3. The default iterator of a dictionary is the keys, but for an array it's the values.

To Do
-----

- Many methods don't work for SampleGame
- Allow casting games down or up in scope EmptyGame -> Game -> SampleGame.
  Currently not possible to save a SampleGame as a Game
- Mixed nash currently keeps a list of all equilibria, and upon finding a new one, iterates through the entire list to determine if any is sufficiently close to the one just found.
  There is probably a faster way to determine uniqueness.
  This isn't really a bottleneck, so it's not that important to fix.
- `subgame.supportset` returns a set representing the support of a profile-like (profile, mixtures, subgame definition).
  It can be useful for making comparisons between supports, e.g. `support_set(x) < support_set(y)` will tell if the support if x is dominated by the support of y.
  This feels like it belongs in a dedicated class instead of as a function, but the class would basically have this one function.
  This function also works for anything that's a mapping of roles to strategies, which means it applies more generally than any specific class.
  Potentially this should just be in `rsgame`.
  It seems like making more of the functions work arbitrarily well on duck-typed things (i.e. mapping of strings to a collection of strings).
- Fix the rest of `gameio`.
- Change conditional in `dominance`, which indicates how to treat missing data to an enum or at least a string
- Some functions in `dominance` could probably be more efficient.
- Incorporate old tests.
- The way a lot of functions handle missing data is not tested very well.
- Low support in a mixture could cause a lot of headaches. Maybe make truncation default?
  Or a global setting somewhere?
- Integrate read the docs with numpy docstyle extension and github travis-ci for testing etc.
- Make nash equilibria methods also return equilibria regret if it's computed.
  Regret is easier to throw away than recalculate.
- Remove static constructors for games, and instead have the constructor choose
  the appropriate one based on type info of passed arguments.
- Get parallel testing working.
  Rednose, which isn't that necessary, seems ti interfere with the parallel module.
  It also seems like generator tests may not be run.
- Replicator dynamics currently has an iteration threshold to terminate at.
  This general only happens if there's a limit cycle, but could occur for any reason.
  One possible way to remove this would be to keep a bloom filter of seen mixtures, after k positive results in a row for testing if we've seen a mixture before we then transition into cycle detection where we run two replicator dynamics at the same time, one at double rate.
  The change will increase computation by 50%, but if we suppress with the bloom filter, that's probably not that bad.
  We could then remove the max_iters parameter.
- If things in collect extend dict or OrderedDict (which extends dict) then they'll be default json serializable, which could clean up some code.
