"""
A setuptools-based setup module.
"""

import os
import re

from setuptools import setup, find_packages
from codecs import open


def read(*names, **kwargs):
    with open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


long_description = read("README.rst")
version = find_version("quantgov", "__init__.py")

setup(
    name='quantgov',
    version=version,

    description='A Policy Analytics Framework',
    long_description=long_description,
    url='https://www.quantgov.org',
    author='Oliver Sherouse',
    author_email='osherouse@mercatus.gmu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='quantgov economics policy government machine learning',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'joblib',
        'pandas',
        'requests',
        'scikit-learn',
        'scipy',
        'snakemake',
    ],
    extras_require={
        'testing': ['pytest-flake8']
    },
    entry_points={
        'console_scripts': [
            'quantgov=quantgov.__main__:main',
        ],
    },
)
