"""
quantgov.estimator.structures

Useful structures for evaluating and training estimators
"""
import collections
import joblib as jl
from sklearn.base import BaseEstimator, TransformerMixin
from six import iteritems
from decorator import decorator
import re

try:
    from spacy.lang.en.stop_words import STOP_WORDS
    from gensim.corpora import Dictionary
    import gensim
    spacy = True
except ImportError:
    spacy = None
    gensim = None


@decorator
def check_spacy(func, *args, **kwargs):
    if spacy is None:
        raise RuntimeError('Must install spacy to use {}'.format(func))
    return func(*args, **kwargs)


@decorator
def check_gensim(func, *args, **kwargs):
    if gensim is None:
        raise RuntimeError('Must install gensim to use {}'.format(func))
    return func(*args, **kwargs)


class _PersistanceMixin(object):
    """
    A Mixin to add a `.save` method to any class that uses joblib to pickle the
    object
    """

    def save(self, path):
        """
        Use joblib to pickle the object.

        Arguments:
            path: an open file object or string holding the path to where the
                object should be saved
        """
        jl.dump(self, path)


class Labels(
    collections.namedtuple('Labels', ['index', 'label_names', 'labels']),
    _PersistanceMixin
):
    """
    A set of labels for training a model.

    Arguments:
        * index: a sequence holding the index values for each document being
            labeled
        * label_names: a sequence holding one name for each label
        * labels: an array-like of label values with
            shape [n_samples x n_labels]
    """
    pass


class Trainers(
    collections.namedtuple('Trainers', ['index', 'vectors']),
    _PersistanceMixin
):
    """
    A set of vectorized documents for training a model

    Arguments:
        * index: a sequence holding the index values for each document
            represented
        * vectors: array-like of document vectors [n_samples x n_features]
    """
    pass


class Model(
    collections.namedtuple('Model', ['label_names', 'model']),
    _PersistanceMixin
):
    """
    A Trained model

    Arguments:
        * label_names: sequence of names for each label the model estimates
        * model: a trained sklearn-like model, implementing `.fit`,
            `.fit_transform`, and `.predict` methods
    """
    pass


class CandidateModel(
    collections.namedtuple('CandidateModel', ['name', 'model', 'parameters'])
):
    """
    A Candidate Model for testing

    Arguments:
        * name: an identifier for this model, unique among candidates
        * model: a trained sklearn-like model, implementing `.fit`,
            `.fit_transform`, and `.predict` methods
        * parameters: a dictionary with parameters names as keys and possible
            parameter values to test as values
    """
    pass


class QGLdaModel(BaseEstimator, TransformerMixin):
    def __init__(self, word_regex=r'\b[A-z]{2,}\b', stop_words=STOP_WORDS):
        self.stop_words = stop_words
        self.word_regex = re.compile(word_regex)

    def transform(self, driver):
        return self.model.transform(driver.stream)

    def create_corpus(self, driver):
        return [self.dictionary.doc2bow([i.group(0)
                for i in self.word_regex.finditer(doc.text)])
                for doc in driver.stream()]

    @check_gensim
    @check_spacy
    def fit(self, driver, alpha=None, eta=None, num_topics=None, passes=None):
        self.dictionary = Dictionary([[i.group(0)
                                      for i in self.word_regex
                                        .finditer(doc.text)]
                                      for doc in driver.stream()])
        stop_ids = [self.dictionary.token2id[stopword] for stopword
                    in self.stop_words if stopword in self.dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in
                    iteritems(self.dictionary.dfs) if docfreq == 1]
        self.dictionary.filter_tokens(stop_ids + once_ids)
        self.corpus = self.create_corpus(driver)
        self.model = gensim.models.ldamodel.LdaModel(self.corpus,
                                                     id2word=self.dictionary,
                                                     alpha=alpha
                                                     eta=eta,
                                                     num_topics=num_topics,
                                                     passes=passes)
        return self
