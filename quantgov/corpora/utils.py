"""
quantgov.corpora.utils - utility functions for the corpus submodule
"""
import sys

from decorator import decorator
from pathlib import Path


def load_driver(corpus):
    corpus = Path(corpus)
    if corpus.name == 'driver.py':
        corpus = corpus.parent
    sys.path.insert(0, str(corpus))
    from driver import driver
    sys.path.pop(0)
    return driver


@decorator
def check_nltk(func, *args, **kwargs):
    if args[-1] is None:
        raise RuntimeError('Must install NLTK to use {}'.format(func))
    return func(*args, **kwargs)


@decorator
def check_textblob(func, *args, **kwargs):
    if args[-2] is None:
        raise RuntimeError('Must install textblob to use {}'.format(func))
    return func(*args, **kwargs)
