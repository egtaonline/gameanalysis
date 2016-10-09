import json
import setuptools


with open('setup.json') as f:
    setuptools.setup(**json.load(f))
