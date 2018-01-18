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
    from gensim.corpora import Dictionary
    from gensim import sklearn_api
    import gensim
except ImportError:
    gensim = None


from sklearn.feature_extraction import stop_words
STOP_WORDS = stop_words.ENGLISH_STOP_WORDS


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


class GensimLda(BaseEstimator, TransformerMixin):
    @check_gensim
    def __init__(self, word_pattern=r'\b[A-z]{2,}\b', stop_words=STOP_WORDS):
        self.stop_words = stop_words
        self.word_pattern = re.compile(word_pattern)

    def transform(self, driver):
        self.test_corpus = self.create_corpus(driver)
        return self.model.transform(self.test_corpus)

    def create_corpus(self, driver):
        return [self.dictionary.doc2bow([i.group(0).lower()
                for i in self.word_pattern.finditer(doc.text)])
                for doc in driver.stream()]

    def fit(self, driver, alpha=None, eta=None, num_topics=1, passes=1):
        self.dictionary = Dictionary([[i.group(0).lower()
                                      for i in self.word_pattern
                                        .finditer(doc.text)
                                       if i not in self.stop_words]
                                      for doc in driver.stream()])
        once_ids = [tokenid for tokenid, docfreq in
                    iteritems(self.dictionary.dfs) if docfreq == 1]
        self.dictionary.filter_tokens(once_ids)
        self.corpus = self.create_corpus(driver)
        self.model = sklearn_api.ldamodel.LdaTransformer(
            alpha=alpha,
            eta=eta,
            num_topics=num_topics,
            passes=passes,
            id2word=self.dictionary
        )
        self.model.fit(self.corpus)
        return self
