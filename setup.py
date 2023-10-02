from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ppi-python",
    version="0.1.2",
    packages=find_packages(),
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
