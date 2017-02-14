import tempfile
import subprocess

import pytest

from pathlib import Path


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        yield tmpdir


@pytest.mark.parametrize('component', ['corpus', 'estimator', 'project'])
def test_download(tmpdir, component):
    comp_dir = tmpdir.joinpath(component)
    subprocess.run(['quantgov', 'start', component, str(comp_dir)])
    assert comp_dir.joinpath('makefile').exists()


def test_noclobber(tmpdir):
    comp_dir = tmpdir.joinpath('corpus')
    comp_dir.mkdir()
    with pytest.raises(subprocess.CalledProcessError)
        subprocess.run(['quantgov', 'start', 'corpus', str(comp_dir)],
                       check=True)
