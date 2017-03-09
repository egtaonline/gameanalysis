import setuptools


setuptools.setup(
    name="gameanalysis",
    version="2.0.0",
    description="A python module for analyzing sparse and empirical games",
    url="https://github.com/egtaonline/gameanalysis.git",
    author="Strategic Reasoning Group",
    author_email='strategic.reasoning.group@umich.edu',
    license="Apache 2.0",
    entry_points={"console_scripts": ["ga=gameanalysis.script:main"]},
    install_requires=[
        "numpy~=1.12",
        "scipy~=0.18",
        "scikit-learn~=0.18",
    ],
    packages=[
        "gameanalysis",
        "gameanalysis.scripts",
    ],
)
