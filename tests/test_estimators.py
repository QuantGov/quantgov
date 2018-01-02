# import pytest
import subprocess
import quantgov.estimator
import quantgov

from pathlib import Path

PSEUDO_CORPUS_PATH = Path(__file__).resolve().parent.joinpath('pseudo_corpus')
driver = quantgov.load_driver(PSEUDO_CORPUS_PATH)
# models = quantgov.estimator.utils.load_models('./sample_models.py')


# def test_all_model_evaluation():
#     quantgov.estimator.evaluation.evaluate_all_models(models, X, y, 2, 'f1')


def test_topic_model():
    sample = quantgov.estimator.structures.QGLdaModel()
    sample.fit(driver)
    sample.transform(driver)


def check_output(cmd):
    return (
        subprocess.check_output(cmd, universal_newlines=True)
        .replace('\n\n', '\n')
    )
