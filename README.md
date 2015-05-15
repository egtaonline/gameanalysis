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
