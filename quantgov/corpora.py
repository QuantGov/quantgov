"""
quantgov.corpora

Utility Classes for Writing QuantGov Corpora
"""

import re
import collections
import csv
import logging
import sys

from collections import namedtuple
from pathlib import Path

from . import utils

log = logging.getLogger(__name__)

Document = namedtuple('Document', ['index', 'text'])


class CorpusStreamer(object):
    """
    A knowledgable wrapper for a CorpusDriver stream
    """
    def __init__(self, iterable):
        self.iterable = iterable
        self.finished = False
        self.index = []

    @property
    def documents_streamed(self):
        return len(self.index)

    def __iter__(self):
        for document in self.iterable:
            self.index.append(document.index)
            yield document
        self.finished = True


class CorpusDriver(object):
    """
    A base class for Corpus Drivers

    This class defines the Corpus Driver interface
    """
    def __init__(self, index_labels):
        if isinstance(index_labels, str):
            index_labels = (index_labels,)
        else:
            try:
                index_labels = tuple(index_labels)
                assert(all(isinstance(i, str) for i in index_labels))
            except (ValueError, AssertionError):
                raise ValueError(
                    "Index Labels must be a string or sequence of strings")
        self.index_labels = index_labels

    def get_streamer(self):
        """
        Return a CorpusStreamer object that wraps this corpus's stream method.
        """
        return CorpusStreamer(self.stream())

    def stream(self):
        """
        Iterate over the corpus

        Return a generator of Document Objects.
        """
        raise NotImplementedError

    def validate_key(self, key):
        if not isinstance(key, collections.Sequence):
            key = tuple(key,)
        if not len(key) == len(self.index_labels):
            raise ValueError("Expected index value of length {}, got length {}"
                             .format(len(self.index_labels), len(key)))

    def __getitem__(self, key):
        self.validate_key(key)
        for idx, text in self.stream():
            if idx == key:
                return text
        raise KeyError("Index value not found: {}".format(key))


class FlatFileCorpusDriver(CorpusDriver):
    """
    Superclass for drivers that keep each document in a separate file.
    """
    def __init__(self, index_labels, encoding="utf-8", cache=True):
        super(FlatFileCorpusDriver, self).__init__(index_labels)
        self.encoding = encoding
        self.cache = cache
        self._mapping = None

    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = {
                idx: path for idx, path in self.gen_indices_and_paths()
            }
        return self._mapping

    def gen_indices_and_paths(self):
        """
        Return an iterator over the indices and paths of the corpus
        """
        raise NotImplementedError

    def read(self, docinfo):
        """
        Given an index and a path, return a Document
        """
        idx, path = docinfo
        log.debug("Reading {}".format(path))
        with path.open(encoding=self.encoding) as inf:
            return Document(idx, inf.read())

    def __getitem__(self, key):
        self.validate_key(key)
        if self.cache:
            path = self.mapping[key]
        else:
            for idx, path in self.gen_indices_and_paths():
                if idx == key:
                    break
            else:
                raise KeyError()
        return self.read((key, path))

    def stream(self):
        return utils.lazy_parallel(self.read, self.gen_indices_and_paths())


class RecursiveDirectoryCorpusDriver(FlatFileCorpusDriver):
    """
    """
    def __init__(self, directory, index_labels, encoding='utf-8', cache=True):
        super(RecursiveDirectoryCorpusDriver, self).__init__(
            index_labels, encoding, cache=cache)
        self.directory = Path(directory).resolve()
        self.encoding = encoding

    def _recursively_gen_indices_and_paths(self, directory=None):
        if directory is None:
            directory = self.directory
        subpaths = sorted(i for i in directory.iterdir()
                          if not i.name.startswith('.'))
        for subpath in subpaths:
            if subpath.is_dir():
                for idx, path in self._recursively_gen_indices_and_paths(
                        subpath):
                    yield (subpath.name,) + idx, path
            else:
                yield (subpath.stem,), subpath

    def gen_indices_and_paths(self):
        return self._recursively_gen_indices_and_paths()


class NamePatternCorpusDriver(FlatFileCorpusDriver):
    """
    Serve a corpus with all files in a single directory and filenames defined
    by a regular expression.

    The index labels are, the group names contained in the regular expression
    in the order that they appear
    """
    def __init__(self, pattern, directory, encoding='utf-8', cache=True):
        self.pattern = re.compile(pattern)
        index_labels = (
            i[0] for i in
            sorted(self.pattern.groupindex.items(), key=lambda x: x[1])
        )
        super(NamePatternCorpusDriver, self).__init__(
            index_labels=index_labels, encoding=encoding, cache=cache)
        self.directory = Path(directory)

    def gen_indices_and_paths(self):
        subpaths = sorted(i for i in self.directory.iterdir()
                          if not i.name.startswith('.'))
        for subpath in subpaths:
            match = self.pattern.search(subpath.stem)
            index = tuple(match.groupdict()[i] for i in self.index_labels)
            yield index, subpath


class IndexDriver(FlatFileCorpusDriver):
    """
    Serve a corpus using an index csv where the final column is the path to the
    file and the other columns form the index. Index label names are taken from
    the csv header.
    """
    def __init__(self, index, encoding='utf-8', cache=True):
        self.index = Path(index)
        with self.index.open(encoding=encoding) as inf:
            index_labels = next(csv.reader(inf))[:-1]
        super(IndexDriver, self).__init__(
            index_labels=index_labels, encoding=encoding, cache=cache)

    def gen_indices_and_paths(self):
        with self.index.open() as inf:
            reader = csv.reader(inf)
            next(reader)
            for row in reader:
                yield tuple(row[:-1]), Path(row[-1])


def load_driver(corpus):
    corpus = Path(corpus)
    if corpus.name == 'driver.py':
        corpus = corpus.parent
    sys.path.insert(0, str(corpus))
    from driver import driver
    sys.path.pop(0)
    return driver
