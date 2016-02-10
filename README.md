Game Analysis
=============

This is a set of python scripts to manipulate empirical game data.


Setup
-----

You can follow the instructions below for how to setup your environment.
Alternatively, if you're using ubuntu you should just be able to execute `make
ubuntu-setup` from this directory to properly setup your environment. You will
need root privileges to properly setup your environment.

Before this library can be used, you need to install several dependencies.

1. Python 3
2. BLAS/LAPACK
3. virtualenv

On ubuntu these dependencies can be installed with:

```
$ sudo apt-get install python3 libatlas-base-dev gfortran
$ sudo pip3 install virtualenv
```

From here, setup virtualenv in this directory, and activate it.

```
$ cd this/directory
$ virtualenv -p python3 .
$ . bin/activate
```

Now install any other python requirements.

```
$ pip3 install -r requirements.txt
```

At this point, you should run the tests to make sure everything was setup properly. Executing `make test` should run all of the tests. If you see something like:

```
. bin/activate && nosetests test
....................................................
----------------------------------------------------------------------
Ran <xxx> tests in <xxx>s

OK
```

Then all of the tests passed!


Usage
-----

After installation, you need to activate virtualenv every time you want to use
this library. It can be activated with `. bin/activate` and deactivated with
`deactivate`.


Testing
-------

All of the tests can be run with `make test`. If you want more fine grained
control, from within the virtualenv you can run `nosetests
test.<unit-test-file-name>[:test-method-name]` to execute only a single test
file, or or only a specific method from within a file. You may also want to add
the option `--nocapture` to output `sysout` and `syserr`, which are usually
captured.

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


Open Design Questions
---------------------

1. Whether to include version numbers or descriptions on profiles / mixtures or keep them a raw description.
2. How to handle games where data doesn't exist for every role strategy pair, but only some agents.
   Currently any incomplete profile is ignored / errors.
   There might not be an efficient way to handle this case.

To Do
-----

- Make Game constructors take almost raw data so that random game can be constructed more efficiently.
  There should also be constructors taking the general sparse text format.
- Make large tests use a flag that can be passed to nosetests and optionally make
- Allow casting games down or up in scope EmptyGame -> Game -> SampleGame.
  Currently not possible to save a SampleGame as a Game
- In rsgame, make `min_payoffs` a hidden member
- In rsgame, make `min_payoffs` and `dev_reps` lazily computed attributes.
  Python allows members to actually be get functions, which can lazily compute something the first time they're referenced.
- Mixed nash currently keeps a list of all equilibria, and upon finding a new one, iterates through the entire list to determine if any is sufficiently close to the one just found.
  There is probably a faster way to determine uniqueness.
  This isn't really a bottleneck, so it's not that important to fix.
- Add python logging to game analysis, and use that for output of various information.
  First application is replicator dynamics information.
- `subgame.supportset` returns a set representing the support of a profile-like (profile, mixtures, subgame definition).
  It can be useful for making comparisons between supports, e.g. `support_set(x) < support_set(y)` will tell if the support if x is dominated by the support of y.
  This feels like it belongs in a dedicated class instead of as a function, but the class would basically have this one function.
  This function also works for anything that's a mapping of roles to strategies, which means it applies more generally than any specific class.
- `SampleGame` make a copy of the input data before calling the super constructor.
  This could probably be avoided, but it would likely require `SampleGame` initializing `Game` values itself.
  That might be a good thing, because currently the payoffs can change between creating a `SampleGame` and calling `remean`, which is very strange behavior.
- Fix the rest of `gameio`.
- Make sure `nash.pure_strategy_regret` properly returns nan if there is missing data.
- Change conditional in `dominance`, which indicates how to treat missing data to an enum or at least a string
- Some functions in `dominance` could probably be more efficient.
- `subgame.subgame` duplicates all of the data in the game, but most of the functions in dominance only rely on the profile map and the strategy sets.
  It'd be much faster to just make a `ShallowSubgame` that points to the original game with updated information.
  This would make calls to dominance on a subgame much faster, but of course, wouldn't work for things like `nash`.
  The best solution is probably making a new class that extends an EmptyGame that has all of the Game methods that don't require scanning over all of the data. This way a reference to the original game can be held without requiring recomputation.
- Incorporate old tests.

Ideas
-----

- Make nash equilibria methods also return equilibria regret if it's computed.
  Regret is easier to throw away than recalculate.
