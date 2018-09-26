import configparser

import sklearn.pipeline

import quantgov.ml


def _autoconvert(value):
    """Convert to int or float if possible, otherwise return string"""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def get_model(modeldefs, configfile):
    """
    Parse config file and configure relevant model
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(configfile)
    models = {i.name: i for i in
              quantgov.ml.utils.load_models(modeldefs)}
    model = models[config['Model']['name']].model
    model.set_params(
        **{i: _autoconvert(j) for i, j in config['Parameters'].items()})
    return model


def train_and_save_model(
        modeldefs,
        configfile,
        vectorizer,
        trainers,
        labels,
        outfile):
    """
    Train and save model described in config file

    Arguments:

        * **modeldefs**:  Path to a python module containing a list of
            `quantgov.ml.CandidateModel` objects in a module-level
            variable named `models'.
        * **configfile**: config file as produced by
            `quantgov ml evaluate`
        * **vectorizer**: an sklearn-compatible Vectorizer object
        * **trainers**: a `quantgov.ml.Trainers` object
        * **labels**: a `quantgov.ml.Labels` object
        * **outfile**: file to which model should be saved
    """
    model = get_model(modeldefs, configfile)
    pipeline = sklearn.pipeline.Pipeline((
        ('vectorizer', vectorizer),
        ('model', model.fit(trainers.vectors, labels.labels)),
    ))
    quantgov.ml.Estimator(labels.label_names, pipeline).save(outfile)
