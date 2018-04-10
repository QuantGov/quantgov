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
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'model.pickle')),
         str(PSEUDO_CORPUS_PATH)]
    )
    assert output == 'file,is_world\ncfr,False\nmoby,False\n'


def test_probability_estimator():
    output = check_output(
        ['quantgov', 'estimator', 'estimate',
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'model.pickle')),
         str(PSEUDO_CORPUS_PATH), '--probability']
    )
    assert output == ('file,is_world_prob\ncfr,0.0899\nmoby,0.0216\n')


def test_probability_estimator_6decimals():
    output = check_output(
        ['quantgov', 'estimator', 'estimate',
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'model.pickle')),
         str(PSEUDO_CORPUS_PATH), '--probability', '--precision', '6']
    )
    assert output == ('file,is_world_prob\ncfr,0.089898\nmoby,0.02162\n')


def test_multiclass_probability_estimator():
    output = check_output(
        ['quantgov', 'estimator', 'estimate',
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'vectorizer.pickle')),
         str(PSEUDO_ESTIMATOR_PATH.joinpath('data', 'modelmulticlass.pickle')),
         str(PSEUDO_CORPUS_PATH), '--probability']
    )
    assert output == ('file,class,probability\n'
                      'cfr,business-and-industry,0.1765\n'
                      'cfr,environment,0.1294\n'
                      'cfr,health-and-public-welfare,0.1785\n'
                      'cfr,money,0.169\n'
                      'cfr,science-and-technology,0.147\n'
                      'cfr,world,0.1997\n'
                      'moby,business-and-industry,0.1804\n'
                      'moby,environment,0.1529\n'
                      'moby,health-and-public-welfare,0.205\n'
                      'moby,money,0.1536\n'
                      'moby,science-and-technology,0.1671\n'
                      'moby,world,0.141\n')
