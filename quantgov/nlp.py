"""
quantgov.nlp: Text-based analysis of documents
"""
import collections
import math
import numpy as np
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
    if not NLTK:
        raise RuntimeError('Must install NLTK to use {}'.format(func))
    return func(*args, **kwargs)


@decorator
def check_textblob(func, *args, **kwargs):
    if not textblob:
        raise RuntimeError('Must install textblob to use {}'.format(func))
    return func(*args, **kwargs)


@decorator
def check_textstat(func, *args, **kwargs):
    if not textstat:
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
        term_pattern = re.compile(pattern.format('|'.join(terms_sorted)))
        term_counts = collections.Counter(
            i.groupdict()['match'] for i in term_pattern.finditer(text)
        )
        if total_label is not None:
            return (
                doc.index
                + tuple(term_counts[i] for i in terms)
                + (sum(term_counts.values()),)
            )
        return (doc.index + tuple(term_counts[i] for i in terms))


commands['count_occurrences'] = OccurrenceCounter


class EnhancedOccurrenceCounter():

    cli = utils.CLISpec(
        help="Term counter, includes bullet points after terms",
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
            ),
            utils.CLIArg(
                flags=('--point_pattern'),
                kwargs={
                    'help': 'pattern to use in identifying bullet points',
                    'default': r'\(.{1,4}\)|^.\.|;$|\d\d?-\d\d?-\d\d?-\d\d?\.'
                }
            ),
            utils.CLIArg(
                flags=('--line_split'),
                kwargs={
                    'help': 'character on which to split lines',
                    'default': '\n'
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
    def process_document(doc, terms, pattern, point_pattern, line_split, total_label):
        terms_sorted = sorted(terms, key=len, reverse=True)
        term_pattern = re.compile(
            pattern.format('|'.join(terms_sorted)), re.IGNORECASE)
        point_pattern = re.compile(point_pattern)
        preamble_pattern = re.compile(r'[:—]$', re.MULTILINE)
        # If no bullet point formatting, run standard analysis
        if (not point_pattern.search(doc.text)
                or not preamble_pattern.search(doc.text)):
            return OccurrenceCounter.process_document(
                doc, terms, pattern, total_label)

        def count_line_terms(line):
            line_term_counts = empty_counter.copy()
            line_term_counts.update(collections.Counter(
                i.groupdict()['match'].lower()
                for i in term_pattern.finditer(line)
            ))
            return np.array(list(line_term_counts.values()))

        def count_preamble_terms(line):
            term_counts = empty_counter.copy()
            extra_term_counts = empty_counter.copy()
            if len(term_pattern.findall(line)) > 1:
                term_counts.update(collections.Counter([
                    term_pattern.findall(line)[-1].lower()
                ]))
                extra_term_counts.update(collections.Counter(
                    i.lower() for i in term_pattern.findall(line)[:-1]
                ))
            else:
                term_counts.update(collections.Counter(
                    i.groupdict()['match'].lower()
                    for i in term_pattern.finditer(line)
                ))
            return (np.array(list(term_counts.values())),
                    np.array(list(extra_term_counts.values())))

        def get_format(text, line=False):
            # (i) (ii) format
            if re.search(r"^\([ivx]+\)", text):
                return r"^\([ivx]+\)"
            # (a) (b) format
            elif re.search(r"^\([a-z]{1,2}\)", text):
                return r"^\([a-z]{1,2}\)"
            # (A) (B) format
            elif re.search(r"^\([A-Z]{1,2}\)", text):
                return r"^\([A-Z]{1,2}\)"
            # (1) (2) format
            elif re.search(r"^\(\d{1,2}\)", text):
                return r"^\(\d{1,2}\)"
            # i. ii. format
            elif re.search(r"^[ivx]+\.", text):
                return r"^[ivx]+\."
            # a. b. format
            elif re.search(r"^[a-hj-r]{1,2}\.", text):
                return r"^[a-hj-r]{1,2}\."
            # A. B. format
            elif re.search(r"^[A-Z]{1,2}\.", text):
                return r"^[A-Z]{1,2}\."
            # 1. 2. format
            elif re.search(r"^\d{1,2}\.", text):
                return r"^\d{1,2}\."
            # none of the above format
            else:
                if line:
                    return ''
                else:
                    return (r"^(?!^\([ivx]{1,4}\)|^\([a-hj-r]{1,2}\)|"
                            r"^\([A-HJ-R]{1,2}\)|^\(\d{1,2}\)|^[ivx]{1,4}|"
                            r"^[a-hj-r]{1,2}\.|^[A-HJ-R]{1,2}\.|^\d{1,2}\.)")

        empty_counter = collections.Counter()
        for t in terms:
            empty_counter.setdefault(t, 0)
        total_count = np.array([0] * len(terms))
        preamble_term_counts = []
        preamble_formatting = []
        line_count = []
        line_formatting = ''
        for line in doc.text.split(line_split):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            if line_formatting and not re.search(
                    line_formatting, line, re.MULTILINE):
                # If no extra lines, count the first line
                if not line_count:
                    line_count.append(1)
                # Multiplies number of keywords found in the preamble
                # by the number of lines in the list
                total_count += line_count[-1] * preamble_term_counts[-1]
                preamble_term_counts.pop()
                line_count.pop()
                # Set nested preamble formatting as new line formatting
                if len(preamble_formatting) > 1:
                    line_formatting = preamble_formatting.pop()
                else:
                    preamble_formatting.pop()
                    line_formatting = ''
            # If the line ends in a ":" or "—" and contains keywords,
            # the line is considered to be a preamble
            if preamble_pattern.search(line):
                preamble_terms, extra_term_counts = count_preamble_terms(line)
                # Add to line count of parent preamble, if nested preamble
                if preamble_formatting:
                    if len(line_count) != len(preamble_formatting):
                        line_count.append(1)
                    else:
                        line_count[-1] += 1
                if preamble_terms.any():
                    preamble_term_counts.append(preamble_terms)
                    preamble_formatting.append(get_format(line))
                    total_count += extra_term_counts
                    line_formatting = ''
            # If bullet pount, add to line count
            # and add any extra terms to total count
            elif get_format(line, line=False) and preamble_formatting:
                if not line_formatting:
                    line_formatting = get_format(line, line=True)
                if len(line_count) != len(preamble_formatting):
                    line_count.append(1)
                else:
                    line_count[-1] += 1
                total_count += count_line_terms(line)
            # For all other lines, add terms to total count
            else:
                total_count += count_line_terms(line)
                # Count preamble terms if not a real list
                if preamble_formatting and not line_formatting:
                    total_count += preamble_term_counts[-1]
                    preamble_term_counts.pop()
                    if len(preamble_formatting) > 1:
                        line_formatting = preamble_formatting.pop()
                    else:
                        preamble_formatting.pop()
        # If leftover counts, add them
        if preamble_formatting:
            if not line_count:
                line_count.append(1)
            total_count += line_count[-1] * preamble_term_counts[-1]
        if total_label:
            return (
                doc.index
                + tuple(total_count)
                + (sum(total_count),)
            )
        return (doc.index + total_count)


commands['enhanced_count_occurrences'] = EnhancedOccurrenceCounter


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
            ),
            utils.CLIArg(
                flags=('--threshold'),
                kwargs={
                    'help': ('maximum average sentence length to allow '
                             '(set to 0 for no filtering)'),
                    'type': int,
                    'default': 100
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
    def process_document(doc, precision, threshold):
        sentences = textblob.TextBlob(doc.text).sentences
        if not len(sentences):
            return doc.index + (None,)
        # Allows for rounding to a specified number of decimals
        elif precision:
            sentence_length = round(sum(len(
                sentence.words) for sentence in sentences) / len(sentences),
                int(precision))
        else:
            sentence_length = sum(len(
                sentence.words) for sentence in sentences) / len(sentences)
        # Filters values based on threshold
        if not threshold or sentence_length < threshold:
            return doc.index + (sentence_length,)
        else:
            return doc.index + (None,)


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
        arguments=[
            utils.CLIArg(
                flags=('--threshold'),
                kwargs={
                    'help': ('minimum score to allow '
                             '(set to 0 for no filtering)'),
                    'type': int,
                    'default': -100
                }
            )
        ]
    )

    @staticmethod
    def get_columns(args):
        return ('flesch_reading_ease',)

    @staticmethod
    @check_textstat
    def process_document(doc, threshold):
        score = textstat.flesch_reading_ease(doc.text)
        # Filters values based on threshold
        if not threshold or score > threshold:
            return doc.index + (int(score),)
        else:
            return doc.index + (None,)


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
