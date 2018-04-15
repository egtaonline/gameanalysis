"""Utilities for game analysis scripts"""
import json
import sys
from collections import abc


def load_profiles(strings):
    """Load profiles from a list of strings

    Parameters
    ----------
    strings : [str]
        A list of strings that are file names or json, and represent either a
        single profile or a list of profiles.

    Returns
    -------
    prof_gen : (prof)
        A generator of json profiles.
    """
    for prof_type in strings:
        # Try to load file else read as string
        if prof_type == '-':
            prof = json.load(sys.stdin)
        else:
            try:
                with open(prof_type) as fil:
                    prof = json.load(fil)
            except FileNotFoundError:
                prof = json.loads(prof_type)
        # Yield different amounts if it's a list
        if isinstance(prof, abc.Mapping):
            yield prof
        else:
            for prf in prof:
                yield prf
