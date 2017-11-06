"""
quantgov.corpora.builtins: Functions for analyzing a single Document
"""
import re
import collections

import quantgov

from nltk.corpus import stopwords
from textblob import Word

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
            )
            quantgov.utils.CLIArg(
                flags=('--stopwords', '-sw'),
                kwargs={
                    'help': 'stopwords to ignore',
                    'default': set(stopwords.words('english'))
                }
            )
        ]

    )

    @staticmethod
    def get_columns(args):
        return ('shannon_entropy',)

    @staticmethod
    def process_document(doc, word_pattern, stopwords):
        words = word_pattern.findall(doc.text)
        lemmas = [
            lemma for lemma in (
                Word(word).lemmatize() for word in words
            )
            if lemma not in stopwords
        ]
        counts = collections.Counter(lemmas)
        return round(sum(
            -(count / len(lemmas) * math.log(count / len(lemmas), 2))
            for count in counts.values()
        ), 2)


commands['shannon_entropy'] = ShannonEntropy


class ConditionalCounter():

    cli = quantgov.utils.CLISpec(
        help='Conditional Counter',
        arguments=[
            quantgov.utils.CLIArg(
                flags=('--conditional_pattern', '-cp'),
                kwargs={
                    'help': 'regular expression defining a "conditional"',
                    'type': re.compile,
                    'default': re.compile(
                        r'\b(if|but|except|provided|when|where|whenever|unless|notwithstanding'
                        r'|in\s+the\s+event|in\s+no\s+event)\b')
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return ('conditional_count',)

    @staticmethod
    def process_document(doc, conditional_pattern):
        return len(pattern.findall(' '.join((doc.text).splitlines())))


commands['conditional_count'] = ConditionalCounter


class SentenceLength():

    cli = quantgov.utils.CLISpec(
        help='Sentence Length',
        arguments=[
            quantgov.utils.CLIArg(
                flags=('--word_pattern', '-wp'),
                kwargs={
                    'help': 'regular expression defining a "word"',
                    'type': re.compile,
                    'default': re.compile(r'\b\w+\b')
                }
            )
            quantgov.utils.CLIArg(
                flags=('--sentence_pattern', '-sp'),
                kwargs={
                    'help': 'regular expression defining a "sentence"',
                    'type': re.compile,
                    'default': re.compile(r'[A-Z][^\.!?]*[\.!?]')
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return ('sentence_length',)

    @staticmethod
    def process_document(doc, word_pattern, sentence_pattern):
        sentences = sentence_pattern.findall(doc)
        total_length = 0
        for sentence in sentences:
            total_length += len(word_pattern.findall(sentence))
        return total_length / len(sentences)


commands['sentence_length'] = SentenceLength