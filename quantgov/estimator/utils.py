import sys

from pathlib import Path


def load_models(path):
    """
    Load models list from path

    Arguments:

        * **path**:  Path to a python module containing a list of
            `quantgov.estimator.CandidateModel` objects in a module-level
    """
    path = Path(path).resolve()
    try:
        assert ' ' not in path.stem
    except AssertionError:
        raise ValueError("models file name must contain no spaces")
    sys.path.insert(0, str(path.parent))
    exec('import {}'.format(path.stem))
    models = eval('{}.models'.format(path.stem))
    exec('del({})'.format(path.stem))
    sys.path.pop(0)
    return models
