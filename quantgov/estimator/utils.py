import sys

from pathlib import Path


def load_models(path):
    """
    Load models list from path

    Arguments:

        * **path**:  Path to a python module containing a list of
            `quantgov.estimator.CandidateModel` objects in a module-level
    """
    path = Path(path)
    sys.path.insert(0, str(path))
    try:
        assert ' ' not in path.name
    except AssertionError:
        raise ValueError("models file name must contain no spaces")
    models = None
    exec('from {} import models'.format(path.name))
    sys.path.pop(0)
    return models
