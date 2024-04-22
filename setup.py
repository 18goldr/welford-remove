from setuptools import setup, find_packages

with open("README.rst") as f:
    readme = f.read()

setup(
    name='welford-remove',
    version=0.1,
    description="Python (numpy) implementation of Welford's algorithm with the ability to remove data points.",
    author="Robert Gold",
    author_email="18goldr@gmail.com",
    url="https://github.com/18goldr/welford-with-remove",
    license=MIT,
    keywords=["statistics", "online", "welford"],
    install_requires=["numpy"],
    long_description=readme,
    packages=find_packages(),

)
