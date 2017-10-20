"""
quantgov.corpora.builtins: Functions for analyzing a single Document
"""
import re

import quantgov


def count_words(doc, word_pattern):
    return doc.index + (len(word_pattern.findall(doc.text)),)


count_words.cli = quantgov.utils.CLISpec(
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
count_words.get_columns = lambda x: ('words',)
