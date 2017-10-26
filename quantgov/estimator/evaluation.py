import configparser
import logging
import sys

import sklearn.model_selection
import pandas as pd

from pathlib import Path

log = logging.log(__name__)


def evaluate_model(model, X, y, folds, scoring):
    log.info('Processing {}'.format(model.name))
    if hasattr(y[0], '__getitem__'):
        cv = sklearn.model_selection.StratifiedKFold(folds, shuffle=True)
        if '_' not in scoring:
            log.warning("No averaging method specified, assuming macro")
            scoring += '_macro'
    else:
        cv = sklearn.model_selection.StratifiedKFold(folds, shuffle=True)
    gs = sklearn.model_selection.GridSearchCV(
        estimator=model.model,
        param_grid=model.parameters,
        cv=cv,
        scoring=scoring,
        verbose=100,
        refit=False
    )
    gs.fit(X, y)
    return pd.DataFrame(gs.cv_results_).assign(model=model.name)


def evaluate_all_models(models, X, y, folds, scoring):
    results = pd.concat([
        evaluate_model(model, X, y, folds, scoring) for model in models])
    results = results[
        ['model', 'mean_test_score', 'std_test_score',
         'mean_train_score', 'std_train_score',
         'mean_fit_time', 'std_fit_time',
         'mean_score_time', 'std_score_time']
        + sorted(i for i in results if i.startswith('param'))
        + sorted(i for i in results if i.startswith('split'))
        + ['params']
    ]
    return results


def write_suggestion(results, file):
    """
    Given results, write the best performer to a config file.

    Arguments:

        * **Results**: a A DataFrame as returned by `evaluate_models`
        * **file**: an open file-like object
    """
    best_model = results.iloc[results['mean_test_score'].idxmax()]
    config = configparser.ConfigParser()
    config.optionxform = str
    config['Model'] = {'name': best_model['model']}
    config['Parameters'] = best_model['params']
    config.write(file)


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


def evaluate_models(modeldefs, labels, trainers, folds, scoring, results_file,
                    suggestion_file):
    """
    Evaluate Candidate Models and write out a suggestion

    Arguments:

        * **modeldefs**:  Path to a python module containing a list of
            `quantgov.estimator.CandidateModel` objects in a module-level
            variable named `models'.
        * **labels**: a `quantgov.estimator.Labels` object
        * **trainers**: a `quantgov.estimator.Trainers` object
        * **folds**: folds to use in cross-validation
        * **results_file**: open file object to which results should be written
        * **suggestion_file**: open file object to which the model suggestion
        should be written
    """
    assert labels.index == trainers.index
    models = load_models(modeldefs)
    results = evaluate_models(
        models, trainers.vectors, labels.labels, folds, scoring)
    results.to_csv(results_file, index=False)
    write_suggestion(results, suggestion_file)
