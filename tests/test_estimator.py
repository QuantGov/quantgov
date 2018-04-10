import pytest
import quantgov.estimator
import subprocess

from pathlib import Path


PSEUDO_CORPUS_PATH = Path(__file__).resolve().parent.joinpath('pseudo_corpus')
PSEUDO_ESTIMATOR_PATH = Path(__file__).resolve().parent.joinpath(
                                                           'pseudo_estimator')


def check_output(cmd):
    return (
        subprocess.check_output(cmd, universal_newlines=True)
        .replace('\n\n', '\n')
    )


def test_simple_estimator():
    output = check_output(
        ['quantgov', 'estimator', 'estimate',
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data/vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data/model.pickle')),
         str(PSEUDO_CORPUS_PATH)]
    )
    assert output == 'file,is_world\n1,False\n2,False\n'


def test_probability_estimator():
    output = check_output(
        ['quantgov', 'estimator', 'estimate',
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data/vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data/model.pickle')),
         str(PSEUDO_CORPUS_PATH), '--probability']
    )
    assert output == ('file,is_world_prob\n1,0.0506\n2,0.034\n')


def test_probability_estimator_6decimals():
    output = check_output(
        ['quantgov', 'estimator', 'estimate',
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data/vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data/model.pickle')),
         str(PSEUDO_CORPUS_PATH), '--probability', '--precision', '6']
    )
    assert output == ('file,is_world_prob\n1,0.050626\n2,0.034038\n')
