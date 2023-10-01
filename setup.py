from setuptools import setup, find_packages

setup(
    name="ppi_py",
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'statsmodels'],
)
