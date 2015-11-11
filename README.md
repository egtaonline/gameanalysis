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

EmptyGame contains several functions that are valid for games without payoff
data.

Game is a potentially sparse mapping from role symmetric profiles to
payoffs. It behaves mostly like a python dictionary from profiles to payoff
dictionaries.

SampleGame retains payoff data for every observation. This allows it to
resample the payoff data for every individual profile.


Profiles
--------

Profiles and Mixtures have a dictionary representation and a corresponding
array representation that is only valid for the existing game object as it
depends on the order that strategy names were hashed. The existence of array
versions is for efficiency of many operations.


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

1. Whether to include version numbers or descriptions on profiles / mixtures or
   keep them a raw description.
2. How to handle games where data doesn't exist for every role strategy pair, but only
   some agents. Currently any incomplete profile is ignored / errors. There
   might not be an efficient way to handle this case.

To Do
-----
1. Have a make target that checks style of all files
2. Make large tests use a flag that can be passed to nosetests and optionally make
3. Make sure scripts are automatically run from the virtualenv
4. Implement SampleGame
5. Potentially move library into its own repository, and have the run scripts
   be in a separate repository.
6. Change individual commands to one command using argparse subparsers
