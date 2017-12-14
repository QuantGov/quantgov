import pytest
import quantgov.corpora
import subprocess

from pathlib import Path


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
        outf.write(u'letter,number,path\n')
        outf.write(u'\n'.join(','.join(row) for row in rows))
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


PSEUDO_CORPUS_PATH = Path(__file__).resolve().parent.joinpath('pseudo_corpus')


def check_output(cmd):
    return (
        subprocess.check_output(cmd, universal_newlines=True)
        .replace('\n\n', '\n')
    )


def test_wordcount():
    output = check_output(
        ['quantgov', 'corpus', 'count_words', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,words\n1,248\n2,800\n'


def test_wordcount_pattern():
    output = check_output(
        ['quantgov', 'corpus', 'count_words', str(PSEUDO_CORPUS_PATH),
         '--word_pattern', '\S+']
    )
    assert output == 'file,words\n1,248\n2,800\n'


def test_termcount():
    output = check_output(
        ['quantgov', 'corpus', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'lorem'],
    )
    assert output == 'file,lorem\n1,1\n2,1\n'


def test_termcount_multiple():
    output = check_output(
        ['quantgov', 'corpus', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'lorem', 'dolor sit'],
    )
    assert output == 'file,lorem,dolor sit\n1,1,1\n2,1,0\n'


def test_termcount_multiple_with_label():
    output = check_output(
        ['quantgov', 'corpus', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'lorem', 'dolor sit', '--total_label', 'bothofem'],
    )
    assert output == 'file,lorem,dolor sit,bothofem\n1,1,1,2\n2,1,0,1\n'


def test_shannon_entropy():
    output = check_output(
        ['quantgov', 'corpus', 'shannon_entropy', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,shannon_entropy\n1,7.14\n2,8.13\n'


def test_shannon_entropy_no_stopwords():
    output = check_output(
        ['quantgov', 'corpus', 'shannon_entropy', str(PSEUDO_CORPUS_PATH),
         '--stopwords', 'None'],
    )
    assert output == 'file,shannon_entropy\n1,7.18\n2,8.09\n'


def test_shannon_entropy_4decimals():
    output = check_output(
        ['quantgov', 'corpus', 'shannon_entropy', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == 'file,shannon_entropy\n1,7.1413\n2,8.1252\n'


def test_conditionalcount():
    output = check_output(
        ['quantgov', 'corpus', 'count_conditionals', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,conditionals\n1,0\n2,0\n'


def test_sentencelength():
    output = check_output(
        ['quantgov', 'corpus', 'sentence_length', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,sentence_length\n1,9.54\n2,8.16\n'


def test_sentencelength_4decimals():
    output = check_output(
        ['quantgov', 'corpus', 'sentence_length', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == 'file,sentence_length\n1,9.5385\n2,8.1633\n'


def test_sentiment_analysis():
    output = check_output(
        ['quantgov', 'corpus', 'sentiment_analysis', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == ('file,sentiment_polarity,sentiment_subjectivity'
                      '\n1,0.0,0.0\n2,0.0,0.0\n')


def test_sentiment_analysis_4decimals():
    output = check_output(
        ['quantgov', 'corpus', 'sentiment_analysis', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == ('file,sentiment_polarity,sentiment_subjectivity'
                      '\n1,0.0,0.0\n2,0.0,0.0\n')


def test_sanity_check():
    output = check_output(
        ['quantgov', 'corpus', 'check_sanity', str(PSEUDO_CORPUS_PATH),
         '--metadata', 'tests/pseudo_corpus/data/metadata.csv']
    )
    assert output == ('BASIC STATISTICS\n'
                      'Number of documents: 2\n'
                      'Total word count: 1048\n'
                      '------------------\n'
                      'EXTREME DOCUMENTS\n'
                      'Largest document by wordcount: 800 words, '
                      'in file data/clean/2.txt\n'
                      'Smallest document by wordcount: 248 words, '
                      'in file data/clean/1.txt\n'
                      'Number of documents with the minimum wordcount: 1\n'
                      '------------------\n'
                      'WARNINGS\n'
                      '>>> WARNING: number of docs with the minimum word count '
                      'is greater than one percent of total corpus! '
                      'Check quality!'
    )


def test_sanity_check_highcutoff():
    output = check_output(
        ['quantgov', 'corpus', 'check_sanity', str(PSEUDO_CORPUS_PATH),
         '--metadata', 'tests/pseudo_corpus/data/metadata.csv',
         '--cutoff', '0.51']
    )
    assert output == ('BASIC STATISTICS\n'
                      'Number of documents: 2\n'
                      'Total word count: 1048\n'
                      '------------------\n'
                      'EXTREME DOCUMENTS\n'
                      'Largest document by wordcount: 800 words, '
                      'in file data/clean/2.txt\n'
                      'Smallest document by wordcount: 248 words, '
                      'in file data/clean/1.txt\n'
                      'Number of documents with the minimum wordcount: 1\n'
                      '------------------\n'
                      'WARNINGS\n'
                      'No warnings to show!'
    )
