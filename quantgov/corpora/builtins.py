"""
quantgov.corpora.builtins: Functions for analyzing a single Document
"""
import re
import collections

import quantgov

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
                    'help': 'list of words to be counted',
                    'default': ['shall', 'must', 'may not',
                                'required', 'prohibited', ],
                    'nargs': '+'
                }
            ),
            quantgov.utils.CLIArg(
                flags=('--pattern'),
                kwargs={
                    'help': 'pattern to use in identifying words',
                    'type': str,
                    'default': r'\b(?P<match>{})\b'
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return (tuple(args['terms']))

    @staticmethod
    def process_document(doc, terms, pattern):
        text = ' '.join(doc.text.split()).lower()
        terms_sorted = sorted(terms, key=len, reverse=True)
        combined_pattern = re.compile(pattern.format('|'.join(terms_sorted)))
        term_counts = collections.Counter(
            i.groupdict()['match'] for i in combined_pattern.finditer(text)
        )
        return doc.index + tuple(term_counts[i] for i in terms)


commands['count_occurrences'] = OccurrenceCounter
