from setuptools import setup

with open("README.rst") as f:
    readme = f.read()

kwargs = {
    "name": "welford-with-remove",
    "version": "1.0",
    "description": "Python (numpy) implementation of Welford's algorithm with the ability to remove data points.",
    "author": "Robert Gold",
    "author_email": "18goldr@gmail.com",
    "url": "https://github.com/18goldr/welford",
    "license": "MIT",
    "keywords": ["statistics", "online", "welford"],
    "install_requires": ["numpy"],
    "packages": ["welford"],
    "long_description": readme,
}

setup(**kwargs)
