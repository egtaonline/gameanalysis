import setuptools


setuptools.setup(
  name="gameanalysis",
  version="1.8",
  description="A python module for analyzing sparse and empirical games",
  url="https://github.com/egtaonline/gameanalysis.git",
  author="Strategic Reasoning Group",
  license="Apache 2.0",
  entry_points=dict(console_scripts=["ga=gameanalysis.script:main"]),
  install_requires=[
    "numpy~=1.11",
    "scipy~=0.18",
    "scikit-learn~=0.18"
  ],
  packages=[
    "gameanalysis",
    "gameanalysis.scripts"
  ],
)
