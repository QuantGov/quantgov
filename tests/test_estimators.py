# import pytest
import subprocess
import quantgov.estimator
import quantgov

from pathlib import Path

PSEUDO_CORPUS_PATH = Path(__file__).resolve().parent.joinpath('pseudo_corpus')
driver = quantgov.load_driver(PSEUDO_CORPUS_PATH)


def test_topic_model():
    sample = quantgov.estimator.structures.GensimLda()
    sample.fit(driver, num_topics=2)
    sample.transform(driver)


def check_output(cmd):
    return (
        subprocess.check_output(cmd, universal_newlines=True)
        .replace('\n\n', '\n')
    )
