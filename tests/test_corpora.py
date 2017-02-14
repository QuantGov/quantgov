import pytest
import quantgov.corpora


def build_recursive_directory_corpus(directory):
    for path, text in (('a/1.txt', u'foo'), ('b/2.txt', u'bar')):
        directory.join(path).write_text(text, encoding='utf-8', ensure=True)
    return quantgov.corpora.RecursiveDirectoryCorpusDriver(
        directory=str(directory), index_labels=('letter', 'number'))


def build_name_pattern_corpus(directory):
    for path, text in (('a_1.txt', u'foo'), ('b_2.txt', u'bar')):
        path = directory.join(path).write_text(
            text, encoding='utf-8', ensure=True)
    return quantgov.corpora.NamePatternCorpusDriver(
        pattern=r'(?P<letter>[a-z])_(?P<number>\d)',
        directory=str(directory)
    )


def build_index_corpus(directory):
    rows = []
    for letter, number, path, text in (
            ('a', '1', 'first.txt', u'foo'),
            ('b', '2', 'second.txt', u'bar')
    ):
        outpath = directory.join(path, abs=1)
        outpath.write_text(text, encoding='utf-8')
        rows.append((letter, number, str(outpath)))
    index_path = directory.join('index.csv')
    with index_path.open('w', encoding='utf-8') as outf:
        outf.write('letter,number,path\n')
        outf.write('\n'.join(','.join(row) for row in rows))
    return quantgov.corpora.IndexDriver(str(index_path))


BUILDERS = {
    'RecursiveDirectoryCorpusDriver': build_recursive_directory_corpus,
    'NamePatternCorpusDriver': build_name_pattern_corpus,
    'IndexDriver': build_index_corpus,
}


@pytest.fixture(scope='module', params=list(BUILDERS.keys()))
def corpus(request, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp(request.param, numbered=True)
    return BUILDERS[request.param](tmpdir)


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
