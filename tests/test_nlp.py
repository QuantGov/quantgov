import pytest
import quantgov.corpus
import subprocess

from pathlib import Path


def build_recursive_directory_corpus(directory):
    for path, text in (('a/1.txt', 'foo'), ('b/2.txt', 'bar')):
        directory.join(path).write_text(text, encoding='utf-8', ensure=True)
    return quantgov.corpus.RecursiveDirectoryCorpusDriver(
        directory=str(directory), index_labels=('letter', 'number'))


def build_name_pattern_corpus(directory):
    for path, text in (('a_1.txt', 'foo'), ('b_2.txt', 'bar')):
        path = directory.join(path).write_text(
            text, encoding='utf-8', ensure=True)
    return quantgov.corpus.NamePatternCorpusDriver(
        pattern=r'(?P<letter>[a-z])_(?P<number>\d)',
        directory=str(directory)
    )


def build_index_corpus(directory):
    rows = []
    for letter, number, path, text in (
            ('a', '1', 'first.txt', 'foo'),
            ('b', '2', 'second.txt', 'bar')
    ):
        outpath = directory.join(path, abs=1)
        outpath.write_text(text, encoding='utf-8')
        rows.append((letter, number, str(outpath)))
    index_path = directory.join('index.csv')
    with index_path.open('w', encoding='utf-8') as outf:
        outf.write('letter,number,path\n')
        outf.write('\n'.join(','.join(row) for row in rows))
    return quantgov.corpus.IndexDriver(str(index_path))


def build_s3_corpus(directory):
    rows = []
    for letter, number, path in (
            ('a', '1', 'quantgov_tests/first.txt'),
            ('b', '2', 'quantgov_tests/second.txt')
    ):
        rows.append((letter, number, path))
    index_path = directory.join('index.csv')
    with index_path.open('w', encoding='utf-8') as outf:
        outf.write('letter,number,path\n')
        outf.write('\n'.join(','.join(row) for row in rows))
    return quantgov.corpus.S3Driver(str(index_path),
                                    bucket='quantgov-databanks')


BUILDERS = {
    'RecursiveDirectoryCorpusDriver': build_recursive_directory_corpus,
    'NamePatternCorpusDriver': build_name_pattern_corpus,
    'IndexDriver': build_index_corpus,
    'S3Driver': build_s3_corpus
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
        ['quantgov', 'nlp', 'count_words', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,words\ncfr,349153\nmoby,216645\n'


def test_wordcount_pattern():
    output = check_output(
        ['quantgov', 'nlp', 'count_words', str(PSEUDO_CORPUS_PATH),
         '--word_pattern', r'\S+']
    )
    assert output == 'file,words\ncfr,333237\nmoby,210130\n'


def test_termcount():
    output = check_output(
        ['quantgov', 'nlp', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'shall'],
    )
    assert output == 'file,shall\ncfr,1946\nmoby,94\n'


def test_termcount_multiple():
    output = check_output(
        ['quantgov', 'nlp', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'shall', 'must', 'may not'],
    )
    assert output == ('file,shall,must,may not\n'
                      'cfr,1946,744,122\nmoby,94,285,5\n')


def test_termcount_multiple_with_label():
    output = check_output(
        ['quantgov', 'nlp', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'shall', 'must', 'may not', '--total_label', 'allofthem'],
    )
    assert output == ('file,shall,must,may not,allofthem\n'
                      'cfr,1946,744,122,2812\nmoby,94,285,5,384\n')


def test_shannon_entropy():
    output = check_output(
        ['quantgov', 'nlp', 'shannon_entropy', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,shannon_entropy\ncfr,10.71\nmoby,11.81\n'


def test_shannon_entropy_no_stopwords():
    output = check_output(
        ['quantgov', 'nlp', 'shannon_entropy', str(PSEUDO_CORPUS_PATH),
         '--stopwords', 'None'],
    )
    assert output == 'file,shannon_entropy\ncfr,9.52\nmoby,10.03\n'


def test_shannon_entropy_4decimals():
    output = check_output(
        ['quantgov', 'nlp', 'shannon_entropy', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == 'file,shannon_entropy\ncfr,10.7127\nmoby,11.813\n'


def test_conditionalcount():
    output = check_output(
        ['quantgov', 'nlp', 'count_conditionals', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,conditionals\ncfr,2132\nmoby,2374\n'


def test_sentencelength():
    output = check_output(
        ['quantgov', 'nlp', 'sentence_length', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,sentence_length\ncfr,18.68\nmoby,25.09\n'


def test_sentencelength_4decimals():
    output = check_output(
        ['quantgov', 'nlp', 'sentence_length', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == 'file,sentence_length\ncfr,18.6828\nmoby,25.0936\n'


def test_sentiment_analysis():
    output = check_output(
        ['quantgov', 'nlp', 'sentiment_analysis', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == ('file,sentiment_polarity,sentiment_subjectivity'
                      '\ncfr,0.01,0.42\nmoby,0.08,0.48\n')


def test_sentiment_analysis_4decimals():
    output = check_output(
        ['quantgov', 'nlp', 'sentiment_analysis', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == ('file,sentiment_polarity,sentiment_subjectivity'
                      '\ncfr,0.0114,0.421\nmoby,0.0816,0.4777\n')
