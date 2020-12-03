"""
quantgov.ml.structures

Useful structures for evaluating and training estimators
"""
import collections
import joblib as jl


class _PersistanceMixin(object):
    """
    A Mixin to add a `.save` method to any class that uses joblib to pickle the
    object
    """

    @classmethod
    def load(cls, path):
        """
        Load a saved object at path `path`
        """
        loaded = jl.load(path)
        if not isinstance(loaded, cls):
            raise ValueError(
                'Expected saved type {}, path {} contained saved type {}'
                .format(cls, path, type(loaded))
            )
        return loaded

    def save(self, path):
        """
        Use joblib to save the object.

        Arguments:
            path: an open file object or string holding the path to where the
                object should be saved
        """
        jl.dump(self, path, compress=True)


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


def is_multiclass(classes):
    """
    Returns True if values in classes are anything but 1, 0, True, or False,
    otherwise returns False.
    """
    try:
        return len(set(int(i) for i in classes) - {0, 1}) != 0
    except ValueError:
        return True


class Estimator(
    collections.namedtuple('Estimator', ['label_names', 'pipeline']),
    _PersistanceMixin
):
    """
    A Trained estimator

    Arguments:
        * label_names: sequence of names for each label the model estimates
        * pipeline: a trained sklearn-like pipeline, implementing `.fit`,
            `.fit_transform`, and `.predict` methods, where the X inputs are a
            sequence of strings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.multilabel = len(self.label_names) > 1
        model = self.pipeline.steps[-1][1]
        if self.multilabel:
            try:
                self.multiclass = any(is_multiclass(i) for i in model.classes_)
            except (AttributeError, TypeError):
                self.multiclass = any(
                    is_multiclass(i.classes_)
                    for i in model.steps[-1][-1].estimators_
                )
        # This allows for pipelines without estimators and classes (Keras)
        else:
            if 'tensor' in str(model):
                self.multiclass = None
            elif model:
                self.multiclass = is_multiclass(model.classes_)
            else:
                self.multiclass = None


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
