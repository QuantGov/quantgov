"""
quantgov.nlp: Text-based analysis of documents
"""
import collections
import math
import re

from decorator import decorator

from . import utils

try:
    import nltk.corpus
    NLTK = True
except ImportError:
    NLTK = None

try:
    import textblob
except ImportError:
    textblob = None

try:
    import textstat
except ImportError:
    textstat = None

if NLTK:
    try:
        nltk.corpus.wordnet.ensure_loaded()
    except LookupError:
        nltk.download('wordnet')
        nltk.corpus.wordnet.ensure_loaded()

commands = {}


@decorator
def check_nltk(func, *args, **kwargs):
    if NLTK is None:
        raise RuntimeError('Must install NLTK to use {}'.format(func))
    return func(*args, **kwargs)


@decorator
def check_textblob(func, *args, **kwargs):
    if textblob is None:
        raise RuntimeError('Must install textblob to use {}'.format(func))
    return func(*args, **kwargs)


@decorator
def check_textstat(func, *args, **kwargs):
    if textstat is None:
        raise RuntimeError('Must install teststat to use {}'.format(func))
    return func(*args, **kwargs)


class WordCounter():

    cli = utils.CLISpec(
        help='Word Counter',
        arguments=[
            utils.CLIArg(
                flags=('--word_pattern', '-wp'),
                kwargs={
                    'help': 'regular expression defining a "word"',
                    'type': re.compile,
                    'default': re.compile(r'\b\w+\b')
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return ('words',)

    @staticmethod
    def process_document(doc, word_pattern):
        return doc.index + (len(word_pattern.findall(doc.text)),)


commands['count_words'] = WordCounter


class OccurrenceCounter():

    cli = utils.CLISpec(
        help="Term Counter for Specific Words",
        arguments=[
            utils.CLIArg(
                flags=('terms'),
                kwargs={
                    'help': 'list of terms to be counted',
                    'nargs': '+'
                }
            ),
            utils.CLIArg(
                flags=('--total_label'),
                kwargs={
                    'metavar': 'LABEL',
                    'help': (
                        'output a column with sum of occurrences of all terms'
                        ' with column name LABEL'
                    ),
                }
            ),
            utils.CLIArg(
                flags=('--pattern'),
                kwargs={
                    'help': 'pattern to use in identifying words',
                    'default': r'\b(?P<match>{})\b'
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        if args['total_label'] is not None:
            return tuple(args['terms']) + (args['total_label'],)
        return tuple(args['terms'])

    @staticmethod
    def process_document(doc, terms, pattern, total_label):
        text = ' '.join(doc.text.split()).lower()
        terms_sorted = sorted(terms, key=len, reverse=True)
        combined_pattern = re.compile(pattern.format('|'.join(terms_sorted)))
        term_counts = collections.Counter(
            i.groupdict()['match'] for i in combined_pattern.finditer(text)
        )
        if total_label is not None:
            return (
                doc.index
                + tuple(term_counts[i] for i in terms)
                + (sum(term_counts.values()),)
            )
        return (doc.index + tuple(term_counts[i] for i in terms))


commands['count_occurrences'] = OccurrenceCounter


class ShannonEntropy():
    lemmas = {}
    cli = utils.CLISpec(
        help='Shannon Entropy',
        arguments=[
            utils.CLIArg(
                flags=('--word_pattern', '-wp'),
                kwargs={
                    'help': 'regular expression defining a "word"',
                    'type': re.compile,
                    'default': re.compile(r'\b\w+\b')
                }
            ),
            utils.CLIArg(
                flags=('--stopwords', '-sw'),
                kwargs={
                    'help': 'stopwords to ignore',
                    'default': (
                        None if not NLTK else
                        nltk.corpus.stopwords.words('english')
                    )
                }
            ),
            utils.CLIArg(
                flags=('--precision'),
                kwargs={
                    'help': 'decimal places to round',
                    'default': 2
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return ('shannon_entropy',)

    @staticmethod
    @check_nltk
    @check_textblob
    def process_document(doc, word_pattern, precision, stopwords,
                         textblob=textblob, nltk=NLTK):
        words = word_pattern.findall(doc.text)
        lemmas = [
            lemma for lemma in (
                ShannonEntropy.lemmatize(word) for word in words
            )
            if lemma not in stopwords
        ]
        counts = collections.Counter(lemmas)
        return doc.index + (round(sum(
            -(count / len(lemmas) * math.log(count / len(lemmas), 2))
            for count in counts.values()
        ), int(precision)),)

    def lemmatize(word):
        if word in ShannonEntropy.lemmas:
            lemma = ShannonEntropy.lemmas[word]
        else:
            lemma = textblob.Word(word).lemmatize()
            ShannonEntropy.lemmas[word] = lemma
        return lemma


commands['shannon_entropy'] = ShannonEntropy


class ConditionalCounter():
    cli = utils.CLISpec(
        help=('Count conditional words and phrases. Included terms are: '
              ' "if", "but", "except", "provided", "when", "where", '
              '"whenever", "unless", "notwithstanding", "in the event", '
              'and "in no event"'),
        arguments=[]
    )
    pattern = re.compile(
        r'\b(if|but|except|provided|when|where'
        r'|whenever|unless|notwithstanding'
        r'|in\s+the\s+event|in\s+no\s+event)\b'
    )

    @staticmethod
    def get_columns(args):
        return ('conditionals',)

    @staticmethod
    def process_document(doc):
        return doc.index + (len(ConditionalCounter.pattern.findall(
                                ' '.join((doc.text).splitlines()))),)


commands['count_conditionals'] = ConditionalCounter


class SentenceLength():

    cli = utils.CLISpec(
        help='Sentence Length',
        arguments=[
            utils.CLIArg(
                flags=('--precision'),
                kwargs={
                    'help': 'decimal places to round',
                    'default': 2
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return ('sentence_length',)

    @staticmethod
    @check_nltk
    @check_textblob
    def process_document(doc, precision):
        sentences = textblob.TextBlob(doc.text).sentences
        # Returns sentence_length = 0 if no complete sentences are found
        if len(sentences) == 0:
            return doc.index + (0,)
        # Allows for rounding to a specified number of decimals
        if precision:
            return doc.index + (round(sum(len(
                sentence.words) for sentence in sentences) / len(sentences),
                int(precision)),)
        else:
            return doc.index + (sum(len(
                sentence.words) for sentence in sentences) / len(sentences),)


commands['sentence_length'] = SentenceLength


class SentimentAnalysis():

    cli = utils.CLISpec(
        help='Performs sentiment analysis on the text',
        arguments=[
            utils.CLIArg(
                flags=('--backend'),
                kwargs={
                    'help': 'which program to use for the analysis',
                    'default': 'textblob'
                }
            ),
            utils.CLIArg(
                flags=('--precision'),
                kwargs={
                    'help': 'decimal places to round',
                    'default': 2
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        if args['backend'] == 'textblob':
            return ('sentiment_polarity', 'sentiment_subjectivity',)
        else:
            raise NotImplementedError

    @staticmethod
    @check_nltk
    @check_textblob
    def process_document(doc, backend, precision):
        if backend == 'textblob':
            sentiment = textblob.TextBlob(doc.text)
            # Allows for rounding to a specified number of decimals
            if precision:
                return (doc.index + (round(
                        sentiment.polarity, int(precision)),
                    round(sentiment.subjectivity, int(precision)),))
            else:
                return (doc.index + (sentiment.polarity,
                                     sentiment.subjectivity,))


commands['sentiment_analysis'] = SentimentAnalysis


class FleschReadingEase():

    cli = utils.CLISpec(
        help='Flesch Reading Ease metric',
        arguments=[]
    )

    @staticmethod
    def get_columns(args):
        return ('flesch_reading_ease',)

    @staticmethod
    @check_textstat
    def process_document(doc):
        score = textstat.flesch_reading_ease(doc.text)
        # Allows for rounding to a specified number of decimals
        return doc.index + (int(score),)


commands['flesch_reading_ease'] = FleschReadingEase


class TextStandard():

    cli = utils.CLISpec(
        help='combines all of the readability metrics in textstats',
        arguments=[]
    )

    @staticmethod
    def get_columns(args):
        return ('text_standard',)

    @staticmethod
    @check_textstat
    def process_document(doc):
        score = textstat.text_standard(doc.text)
        # Allows for rounding to a specified number of decimals
        return doc.index + (score,)


commands['text_standard'] = TextStandard
