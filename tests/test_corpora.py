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
    assert output == 'file,words\ncfr,349153\nmoby,216645\n'


def test_wordcount_pattern():
    output = check_output(
        ['quantgov', 'corpus', 'count_words', str(PSEUDO_CORPUS_PATH),
         '--word_pattern', '\S+']
    )
    assert output == 'file,words\ncfr,333237\nmoby,210130\n'


def test_termcount():
    output = check_output(
        ['quantgov', 'corpus', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'shall'],
    )
    assert output == 'file,shall\ncfr,1946\nmoby,94\n'


def test_termcount_multiple():
    output = check_output(
        ['quantgov', 'corpus', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'shall', 'must', 'may not'],
    )
    assert output == ('file,shall,must,may not\n'
                      'cfr,1946,744,122\nmoby,94,285,5\n')


def test_termcount_multiple_with_label():
    output = check_output(
        ['quantgov', 'corpus', 'count_occurrences', str(PSEUDO_CORPUS_PATH),
         'shall', 'must', 'may not', '--total_label', 'allofthem'],
    )
    assert output == ('file,shall,must,may not,allofthem\n'
                      'cfr,1946,744,122,2812\nmoby,94,285,5,384\n')


def test_shannon_entropy():
    output = check_output(
        ['quantgov', 'corpus', 'shannon_entropy', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,shannon_entropy\ncfr,10.71\nmoby,11.81\n'


def test_shannon_entropy_no_stopwords():
    output = check_output(
        ['quantgov', 'corpus', 'shannon_entropy', str(PSEUDO_CORPUS_PATH),
         '--stopwords', 'None'],
    )
    assert output == 'file,shannon_entropy\ncfr,9.52\nmoby,10.03\n'


def test_shannon_entropy_4decimals():
    output = check_output(
        ['quantgov', 'corpus', 'shannon_entropy', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == 'file,shannon_entropy\ncfr,10.7127\nmoby,11.813\n'


def test_conditionalcount():
    output = check_output(
        ['quantgov', 'corpus', 'count_conditionals', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,conditionals\ncfr,2132\nmoby,2374\n'


def test_sentencelength():
    output = check_output(
        ['quantgov', 'corpus', 'sentence_length', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == 'file,sentence_length\ncfr,18.68\nmoby,25.09\n'


def test_sentencelength_4decimals():
    output = check_output(
        ['quantgov', 'corpus', 'sentence_length', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == 'file,sentence_length\ncfr,18.6827\nmoby,25.0936\n'


def test_sentiment_analysis():
    output = check_output(
        ['quantgov', 'corpus', 'sentiment_analysis', str(PSEUDO_CORPUS_PATH)],
    )
    assert output == ('file,sentiment_polarity,sentiment_subjectivity'
                      '\ncfr,0.01,0.42\nmoby,0.08,0.48\n')


def test_sentiment_analysis_4decimals():
    output = check_output(
        ['quantgov', 'corpus', 'sentiment_analysis', str(PSEUDO_CORPUS_PATH),
         '--precision', '4'],
    )
    assert output == ('file,sentiment_polarity,sentiment_subjectivity'
                      '\ncfr,0.0114,0.421\nmoby,0.0816,0.4777\n')


def test_sanity_check():
    output = check_output(
        ['quantgov', 'corpus', 'check_sanity', str(PSEUDO_CORPUS_PATH),
         '--metadata', 'tests/pseudo_corpus/data/metadata.csv']
    )
    assert output == ('There are 2 documents, for a total word count of '
                      '565,798.\n'
                      'The biggest document is cfr.txt, with a word count of '
                      '349,153.\n'
                      'The smallest document is moby.txt, with a word count of '
                      '216,645. There are 1 of these documents.\n'
                      'WARNING: Number of docs with the minimum word '
                      'count is greater than the allowed proportion. '
                      'Check quality.'
    )


def test_sanity_check_highcutoff():
    output = check_output(
        ['quantgov', 'corpus', 'check_sanity', str(PSEUDO_CORPUS_PATH),
         '--metadata', 'tests/pseudo_corpus/data/metadata.csv',
         '--cutoff', '0.51']
    )
    assert output == ('There are 2 documents, for a total word count of '
                      '565,798.\n'
                      'The biggest document is cfr.txt, with a word count of '
                      '349,153.\n'
                      'The smallest document is moby.txt, with a word count of '
                      '216,645. There are 1 of these documents.\n'
                      'No warnings to show.'
    )
