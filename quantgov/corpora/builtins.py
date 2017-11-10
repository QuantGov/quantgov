"""
quantgov.corpora.builtins: Functions for analyzing a single Document
"""
import re
import collections
import math

import quantgov

import nltk.corpus
import textblob

commands = {}


class WordCounter():

    cli = quantgov.utils.CLISpec(
        help='Word Counter',
        arguments=[
            quantgov.utils.CLIArg(
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

    cli = quantgov.utils.CLISpec(
        help="Term Counter for Specific Words",
        arguments=[
            quantgov.utils.CLIArg(
                flags=('terms'),
                kwargs={
                    'help': 'list of terms to be counted',
                    'nargs': '+'
                }
            ),
            quantgov.utils.CLIArg(
                flags=('--total_label'),
                kwargs={
                    'metavar': 'LABEL',
                    'help': (
                        'output a column with sum of occurrences of all terms'
                        ' with column name LABEL'
                    ),
                }
            ),
            quantgov.utils.CLIArg(
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
    cli = quantgov.utils.CLISpec(
        help='Shannon Entropy',
        arguments=[
            quantgov.utils.CLIArg(
                flags=('--word_pattern', '-wp'),
                kwargs={
                    'help': 'regular expression defining a "word"',
                    'type': re.compile,
                    'default': re.compile(r'\b\w+\b')
                }
            ),
            quantgov.utils.CLIArg(
                flags=('--stopwords', '-sw'),
                kwargs={
                    'help': 'stopwords to ignore',
                    'default': nltk.corpus.stopwords.words('english')
                }
            ),
            quantgov.utils.CLIArg(
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
    def process_document(doc, word_pattern, stopwords, precision):
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
    cli = quantgov.utils.CLISpec(
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

    cli = quantgov.utils.CLISpec(
        help='Sentence Length',
        arguments=[
            quantgov.utils.CLIArg(
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
    def process_document(doc, precision):
        sentences = textblob.TextBlob(doc.text).sentences
        return doc.index + (round(sum(len(
            sentence.words) for sentence in sentences) /
            len(sentences), int(precision)),)


commands['sentence_length'] = SentenceLength
