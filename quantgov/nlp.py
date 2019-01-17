"""
quantgov.nlp: Text-based analysis of documents
"""

from __future__ import division

import re
import collections
import math

from decorator import decorator

from . import utils

try:
    import nltk.corpus
    from nltk import word_tokenize, sent_tokenize, bigrams, trigrams, pos_tag
    NLTK = True
except ImportError:
    NLTK = None

try:
    import textblob
except ImportError:
    textblob = None

if NLTK:
    try:
        nltk.corpus.wordnet.ensure_loaded()
    except LookupError:
        nltk.download('wordnet')
        nltk.corpus.wordnet.ensure_loaded()
    try:
        nltk.pos_tag('A test.')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

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


class WordCounter:

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
        return ('words', )

    @staticmethod
    def process_document(doc, word_pattern):
        return doc.index + (len(word_pattern.findall(doc.text)),)


commands['count_words'] = WordCounter


class OccurrenceCounter:

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


class ShannonEntropy:
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


class ConditionalCounter:
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


class SentenceLength:

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
        # Allows for rounding to a specified number of decimals
        if precision:
            return doc.index + (round(sum(len(
                sentence.words) for sentence in sentences) /
                len(sentences), int(precision)),)
        else:
            return doc.index + (sum(len(
                sentence.words) for sentence in sentences) /
                len(sentences),)


commands['sentence_length'] = SentenceLength


class SentimentAnalysis:

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
                return (doc.index +
                        (round(sentiment.polarity, int(precision)),
                            round(sentiment.subjectivity, int(precision)),))
            else:
                return (doc.index +
                        (sentiment.polarity, sentiment.subjectivity,))


commands['sentiment_analysis'] = SentimentAnalysis


class DesignWords:

    cli = utils.CLISpec(
        help='Searches for a pre-defined list of words potentially '
             'associated with design-based standards in text.',
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
        # column names to return
        return ('design_word_count', 'design_word_ratio',
                'design_word_ratio2', )

    @staticmethod
    @check_nltk
    def process_document(doc, precision):

        # load in design words
        # aka weights and measures, chemical compounds, etc.
        design_words = []
        with open("quantgov/resources/design_words.txt", 'r') as d:
            for l in d:
                design_words.append(l.strip())
        design_words = [x.lower().strip()
                        for x in design_words if x != ""]

        # kill stopwords
        stw = set(nltk.corpus.stopwords.words('english'))
        design_words = [x for x in design_words if x not in stw]

        # 1-3 grams in design words list
        dw1 = set([x for x in design_words
                   if len(word_tokenize(x)) == 1])
        dw2 = set([x for x in design_words
                   if len(word_tokenize(x)) == 2])
        dw3 = set([x for x in design_words
                   if len(word_tokenize(x)) == 3])

        # tokenize document
        tokenized = word_tokenize(doc.text)

        # silly count based on words that might indicate design standards
        # aka best practices, etc.
        maybe_relevant_count = len([x for x in tokenized
                                    if x in ['standard',
                                             'practice',
                                             'best practice']])

        # single words
        token_count = collections.Counter(tokenized)
        dw1_count = sum([token_count[x]
                         for x in token_count.keys() if x in dw1])

        # bigrams, trigrams
        bigrams = [' '.join(x) for x in nltk.bigrams(tokenized)]
        trigrams = [' '.join(x) for x in nltk.trigrams(tokenized)]
        bigrams = collections.Counter(bigrams)
        trigrams = collections.Counter(trigrams)
        dw2_count = sum([bigrams[x]
                         for x in bigrams.keys() if x in dw2])
        dw3_count = sum([trigrams[x]
                         for x in trigrams.keys() if x in dw3])

        # final counts
        design_word_count = dw1_count + dw2_count + dw3_count
        design_word_ratio = design_word_count / sum(token_count.values())
        design_word_ratio2 = design_word_count / len(set(tokenized))

        # rounds
        if precision:
            design_word_ratio = round(design_word_ratio, precision)
            design_word_ratio2 = round(design_word_ratio2, precision)

        return doc.index + (design_word_count, design_word_ratio,
                            design_word_ratio2, )


commands['design_words'] = DesignWords


class PartsOfSpeech:

    cli = utils.CLISpec(
        help='Part of speech tagging and derived metrics',
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
        # column names to return
        return ('', '', )

    @staticmethod
    @check_nltk
    def process_document(doc, precision):

        # NLTK part of speech tagging
        nltk_tags = pos_tag(word_tokenize(doc.text))

        # all tags
        all_tags = []
        with open('quantgov/resources/nltk_pos_tags.txt', 'r') as o:
            for x in o.readlines():
                all_tags.append(x.split('|')[0])

        # count up tags
        count_tags = {}
        for x in all_tags:
            count_tags[x.strip()] = 0
        for x in nltk_tags:
            try:
                count_tags[x[1]] += 1
            except KeyError:
                continue

        word_count = sum(count_tags.values())
        nouns_count = (count_tags['NN'] + count_tags['NNS'] +
                       count_tags['NNP'])
        verbs_count = (count_tags['VB'] + count_tags['VBD'] +
                       count_tags['VBG'] + count_tags['VBN'] +
                       count_tags['VBP'] + count_tags['VBZ'])
        noun_verb_ratio = (nouns_count + 1) / (verbs_count + 1)
        nouns_ratio = (nouns_count + 1) / (word_count + 1)
        verbs_ratio = (verbs_count + 1) / (word_count + 1)
        proper_nouns_count = count_tags['NNP'] + count_tags['NNPS']
        proper_nouns_ratio = (proper_nouns_count + 1) / (word_count + 1)

        if precision:
            noun_verb_ratio = round(noun_verb_ratio, precision)
            nouns_ratio = round(nouns_ratio, precision)
            proper_nouns_ratio = round(proper_nouns_ratio, precision)
            verbs_ratio = round(verbs_ratio, precision)

        return (doc.index +
                (noun_verb_ratio, nouns_count, verbs_count,
                 nouns_ratio, verbs_ratio,
                 proper_nouns_count, proper_nouns_ratio, ))


commands['pos_metrics'] = PartsOfSpeech
