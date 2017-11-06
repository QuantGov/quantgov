#!/usr/bin/env python
import argparse
import csv
import collections
import concurrent.futures
import io
import math
import logging
import sys
import re

from nltk.corpus import stopwords

from textblob import Word
#
from pathlib import Path

#
ENCODE_IN = 'utf-8'
ENCODE_OUT = 'utf-8'


CYCLOMATICS = re.compile(
    r'\b(if|but|except|provided|when|where|whenever|unless|notwithstanding'
    r'|in\s+the\s+event|in\s+no\s+event)\b'
)

WORDS = re.compile(r'\b\w+\b')

LEMMAS = {}
STOPWORDS = set(stopwords.words('english'))

log = logging.getLogger(Path(__file__).stem)


def lemmatize(word):
    if word in LEMMAS:
        lemma = LEMMAS[word]
    else:
        lemma = Word(word).lemmatize()
        LEMMAS[word] = lemma
    return lemma


def count_cyclomatics(text):
    return len(CYCLOMATICS.findall(' '.join(text.splitlines())))


def get_shannon_entropy(text, words):
    lemmas = [
        lemma for lemma in (
            lemmatize(word) for word in words
        )
        if lemma not in STOPWORDS
    ]
    counts = collections.Counter(lemmas)
    return round(sum(
        -(count / len(lemmas) * math.log(count / len(lemmas), 2))
        for count in counts.values()
    ), 2)


def get_row_for_file(path):
    text = path.read_text(encoding=ENCODE_IN).lower()
    file = path.stem
    words = WORDS.findall(text)
    return (
        file, len(words), len(set(words)), count_cyclomatics(text),
        get_shannon_entropy(text, words)
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('indir', type=Path)
    parser.add_argument('-o', '--outfile',
                        type=lambda x: open(
                            x, 'w', newline='', encoding=ENCODE_OUT),
                        default=io.TextIOWrapper(
                            sys.stdout.buffer, encoding=ENCODE_OUT)
                        )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_const',
                           const=logging.DEBUG, default=logging.INFO)
    verbosity.add_argument('-q', '--quiet', dest='verbose',
                           action='store_const', const=logging.WARNING)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    writer = csv.writer(args.outfile)
    writer.writerow(
        ('file', 'words', 'unique words',
         'cyclomatic_complexity', 'shannon_entropy')
    )
    with concurrent.futures.ProcessPoolExecutor() as pool:
        for file, words, uniques, cyclo, entropy in pool.map(
            get_row_for_file, args.indir.iterdir()
        ):
            log.info(f'finished {file}')
            writer.writerow((file, words, uniques, cyclo, entropy))


if __name__ == "__main__":
    main()
