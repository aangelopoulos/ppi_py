from setuptools import setup, find_packages
from pathlib import Path

long_description = open("./assets/PyPI-README.md").read()

setup(
    name="ppi-python",
    version="0.1.4",
    packages=find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
        "scikit-learn",
        "gdown",
    ],
)
