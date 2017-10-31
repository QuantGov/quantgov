import configparser

import quantgov.estimator


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
              quantgov.estimator.utils.load_models(modeldefs)}
    model = models[config['Model']['name']].model
    model.set_params(
        **{i: _autoconvert(j) for i, j in config['Parameters'].items()})
    return model


def train_and_save_model(modeldefs, configfile, trainers, labels, outfile):
    """
    Train and save model described in config file

    Arguments:

        * **modeldefs**:  Path to a python module containing a list of
            `quantgov.estimator.CandidateModel` objects in a module-level
            variable named `models'.
        * **configfile**: config file as produced by
            `quantgov estimator evaluate`
        * **trainers**: a `quantgov.estimator.Trainers` object
        * **labels**: a `quantgov.estimator.Labels` object
        * **outfile**: file to which model should be saved
    """

    model = get_model(modeldefs, configfile)
    model.fit(trainers.vectors, labels.labels)
    quantgov.estimator.Model(labels.label_names, model).save(outfile)
