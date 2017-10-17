"""
A setuptools-based setup module.
"""

import os
import re

from setuptools import setup
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

    description='A Machine-Learning Framework for Turning Text Into Data',
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='quantgov economics policy government machine learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['quantgov'],
    install_requires=[
        'requests',
        'futures;python_version<"3.2"',
        'pathlib;python_version<"3.4"',
    ],
    extras_require={
        'testing': ['pytest-flake8']
    },
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'quantgov=quantgov.__main__:main',
        ],
    },
)
