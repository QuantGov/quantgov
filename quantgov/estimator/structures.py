"""
quantgov.estimator.structures

Useful structures for evaluating and training estimators
"""
import collections
import joblib as jl


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
