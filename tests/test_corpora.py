import tempfile

import pytest
import quantgov.corpora

from pathlib import Path


def build_recursive_directory_corpus(directory):
    for path, text in (('a/1.txt', 'foo'), ('b/2.txt', 'bar')):
        path = directory.joinpath(path)
        try:
            path.parent.mkdir(parents=True)
        except FileExistsError:
            continue
        with path.open('w', encoding='utf-8') as outf:
            outf.write(text)
    return quantgov.corpora.RecursiveDirectoryCorpusDriver(
        directory=str(directory), index_labels=('letter', 'number'))


def build_name_pattern_corpus(directory):
    for path, text in (('a_1.txt', 'foo'), ('b_2.txt', 'bar')):
        path = directory.joinpath(path)
        with path.open('w', encoding='utf-8') as outf:
            outf.write(text)
    return quantgov.corpora.NamePatternCorpusDriver(
        pattern=r'(?P<letter>[a-z])_(?P<number>\d)',
        directory=directory
    )


def build_index_corpus(directory):
    rows = []
    for letter, number, path, text in (
            ('a', '1', 'first.txt', 'foo'),
            ('b', '2', 'second.txt', 'bar')
    ):
        outpath = directory.joinpath(path)
        with outpath.open('w', encoding='utf-8') as outf:
            outf.write(text)
        rows.append((letter, number, str(outpath.resolve())))
    index_path = directory.joinpath('index.csv')
    with index_path.open('w', encoding='utf-8') as outf:
        outf.write('letter,number,path\n')
        outf.write('\n'.join(','.join(row) for row in rows))
    return quantgov.corpora.IndexDriver(index_path)


BUILDERS = {
    'RecursiveDirectoryCorpusDriver': build_recursive_directory_corpus,
    'NamePatternCorpusDriver': build_name_pattern_corpus,
    'IndexDriver': build_index_corpus,
}


@pytest.fixture(scope='module', params=list(BUILDERS.keys()))
def corpus(request):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        yield BUILDERS[request.param](tmpdir)


def test_index_labels(corpus):
    assert corpus.index_labels == ('letter', 'number')


def test_simple_stream(corpus):
    served = tuple(corpus.stream())
    assert served == (
        (('a', '1'), 'foo'),
        (('b', '2'), 'bar')
    )


def test_corpus_streamer(corpus):
    streamer = corpus.get_streamer()
    served = []
    for i in streamer:
        served.append(i)
        assert streamer.documents_streamed == len(served)
        assert not streamer.finished
    assert streamer.documents_streamed == len(served)
    assert streamer.finished 
    assert tuple(served) == (
        (('a', '1'), 'foo'),
        (('b', '2'), 'bar')
    )
    assert streamer.index == [('a', '1'), ('b', '2')]
