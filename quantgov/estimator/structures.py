import collections
import joblib as jl


class _PersistanceMixin(object):
    def save(self, path):
        jl.dump(self, path)


class Labels(
    collections.namedtuple('Labels', ['index', 'label_names', 'labels']),
    _PersistanceMixin
):
    pass


class Trainers(
    collections.namedtuple('Trainers', ['index', 'vectors']),
    _PersistanceMixin
):
    pass


class Model(
    collections.namedtuple('Model', ['label_names', 'model']),
    _PersistanceMixin
):
    pass


class CandidateModel(
    collections.namedtuple('CandidateModel', ['name', 'model', 'parameters'])
):
    pass
