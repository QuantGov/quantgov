import configparser
import logging

import sklearn.model_selection
import pandas as pd

from . import utils as eutils

log = logging.getLogger(name=__name__)


def evaluate_model(model, X, y, folds, scoring):
    """
    Evaluate a single model

    Arguments:
        * model: a quantgov.estimator.Model
        * X: array-like of document vectors with shape [n_samples x n_features]
        * y: array-like of labels with shape [n_samples X n_labels]
        * folds: folds to use in cross-validation
        * scoring: scoring method


    Returns: pandas DataFrame with model evaluation results
    """
    log.info('Evaluating {}'.format(model.name))
    if hasattr(y[0], '__getitem__'):
        cv = sklearn.model_selection.KFold(folds, shuffle=True)
        if '_' not in scoring:
            log.warning("No averaging method specified, assuming macro")
            scoring += '_macro'
    else:
        cv = sklearn.model_selection.KFold(folds, shuffle=True)
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
    """
    Evaluate a number of models

    Arguments:
        * models: a sequence of quantgov.estimator.Model objects
        * X: array-like of document vectors with shape [n_samples x n_features]
        * y: array-like of labels with shape [n_samples X n_labels]
        * folds: folds to use in cross-validation
        * scoring: scoring method

    Returns: pandas DataFrame with model evaluation results
    """
    results = pd.concat(
        [evaluate_model(model, X, y, folds, scoring) for model in models],
        ignore_index=True
    )
    results = results[
        ['model', 'mean_test_score', 'std_test_score',
         'mean_fit_time', 'std_fit_time',
         'mean_score_time', 'std_score_time']
        + sorted(i for i in results if i.startswith('param_'))
        + sorted(i for i in results
                 if i.startswith('split')
                 and '_train_' not in i
                 )
        + ['params']
    ]
    return results


def write_suggestion(results, file):
    """
    Given results, write the best performer to a config file.

    Arguments:

        * **Results**: a A DataFrame as returned by `evaluate_all_models`
        * **file**: an open file-like object
    """
    best_model = results.loc[results['mean_test_score'].idxmax()]
    config = configparser.ConfigParser()
    config.optionxform = str
    config['Model'] = {'name': best_model['model']}
    config['Parameters'] = {i: j for i, j in best_model['params'].items()}
    config.write(file)


def evaluate(modeldefs, trainers, labels, folds, scoring, results_file,
             suggestion_file):
    """
    Evaluate Candidate Models and write out a suggestion

    Arguments:

        * **modeldefs**:  Path to a python module containing a list of
            `quantgov.estimator.CandidateModel` objects in a module-level
            variable named `models'.
        * **trainers**: a `quantgov.estimator.Trainers` object
        * **labels**: a `quantgov.estimator.Labels` object
        * **folds**: folds to use in cross-validation
        * **scoring**: scoring method to use
        * **results_file**: open file object to which results should be written
        * **suggestion_file**: open file object to which the model suggestion
        should be written
    """
    assert labels.index == trainers.index
    models = eutils.load_models(modeldefs)
    results = evaluate_all_models(
        models, trainers.vectors, labels.labels, folds, scoring)
    results.to_csv(results_file, index=False)
    write_suggestion(results, suggestion_file)
