from setuptools import setup, find_packages

setup(
    name="ppi-python",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "statsmodels", "scikit-learn", "gdown"],
)
