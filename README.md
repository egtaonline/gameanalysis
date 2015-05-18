Game Analysis
=============

This is a set of python scripts to manipulate empirical game data.


Installation
------------

Before this library can be used, you need to install several dependencies.

1. Python 3
2. BLAS/LAPACK
3. virtualenv

On ubuntu these dependencies can be installed with:

```
$ sudo apt-get install python3 libatlas-base-dev
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
$ pip install -r requirements.txt
```

TODO: At some point run tests to check.


Usage
-----

After installation, you need to activate virtualenv every time you want to use
this library. It can be activated with `. bin/activate` and deactivated with
`deactivate`.


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

1. Are profiles and mixed profiles necessary, or should we just use arbitrary
   mapping types and have functions that can operate on them?
2. Should the array representation of profiles be exposed or hidden?
3. Figure out exactly what the interface of Game is. Specifically, for
   efficiency of some calculations it's necessary to expose things like _mask,
   _counts, and _value, as well as the array form of some profiles for
   efficiency. However, they're unintuitive, and so it seems weird to expose
   them.
4. PureProfile and MixedProfile or Profile and Mixture
5. Whether to include version numbers or descriptions on profiles / mixtures or
   keep them a raw description.
6. Need to change defualt game representation from
   [{role: [(strat, count, payoff)]}] to [{role: [(strat, count, [payoffs])]}]
7. Handle games where data doesn't exist for every role strategy pair, but only
   some agents.
