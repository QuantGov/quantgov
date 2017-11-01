import subprocess

import pytest


@pytest.mark.parametrize('component', ['corpus', 'estimator', 'project'])
def test_download(tmpdir, component):
    comp_dir = tmpdir.join(component)
    subprocess.check_call(['quantgov', 'start', component, str(comp_dir)])
    assert comp_dir.join('Snakefile').check()


def test_noclobber(tmpdir):
    comp_dir = tmpdir.join('corpus')
    comp_dir.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(['quantgov', 'start', 'corpus', str(comp_dir)])
