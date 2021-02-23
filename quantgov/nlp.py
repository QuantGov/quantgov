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
    import spacy
    from spacy import displacy
    from pysbd.utils import PySBDFactory
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 8000000
    spacy = True
except ImportError:
    spacy = None

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
def check_spacy(func, *args, **kwargs):
    if not spacy:
        raise RuntimeError('Must install spacy to use {}'.format(func))
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
        if args['total_label']:
            return tuple(args['terms']) + (args['total_label'],)
        return tuple(args['terms'])

    @staticmethod
    def process_document(doc, terms, pattern, point_pattern, line_split, total_label):
        terms_sorted = sorted(terms, key=len, reverse=True)
        term_pattern = re.compile(
            pattern.format('|'.join(terms_sorted)), re.IGNORECASE)
        point_pattern = re.compile(point_pattern)
        preamble_pattern_multiline = re.compile(r'[:—-]$', re.MULTILINE)
        preamble_pattern = re.compile(r'[:—-]$')
        # If no bullet point formatting, run standard analysis
        if (not point_pattern.search(doc.text)
                and not preamble_pattern_multiline.search(doc.text)):
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
            preamble = line.split('. ')
            # More than one sentence and last sentence contains a term
            if len(preamble) > 1 and term_pattern.findall(preamble[-1]):
                term_counts.update(collections.Counter([
                    term_pattern.findall(preamble[-1])[-1].lower()
                ]))
                extra_term_counts.update(collections.Counter(
                    i.lower() for i in term_pattern.findall(line)[:-1]
                ))
            # More than one sentence but last sentence contains no terms
            elif len(preamble) > 1 and not term_pattern.findall(preamble[-1]):
                extra_term_counts.update(collections.Counter(
                    i.lower() for i in term_pattern.findall(line)
                ))
            # One sentence but more than one term
            elif len(term_pattern.findall(line)) > 1:
                term_counts.update(collections.Counter([
                    term_pattern.findall(line)[-1].lower()
                ]))
                extra_term_counts.update(collections.Counter(
                    i.lower() for i in term_pattern.findall(line)[:-1]
                ))
            # One sentence and only one term
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
                preamble_term_counts.append(preamble_terms)
                preamble_formatting.append(get_format(line))
                total_count += extra_term_counts
                line_formatting = ''
                line_count.append(0)
            # If bullet pount, add to line count
            # and add any extra terms to total count
            elif get_format(line, line=True) and preamble_formatting:
                if not line_formatting:
                    line_formatting = get_format(line, line=True)
                line_count[-1] += 1
                # Only count terms in line if different from preamble
                if (count_line_terms(line) != preamble_term_counts[-1]).any():
                    total_count += count_line_terms(line)
            # For all other lines, add terms to total count
            else:
                total_count += count_line_terms(line)
                # Count preamble terms if not a real list
                if preamble_formatting and not line_formatting:
                    total_count += preamble_term_counts[-1]
                    preamble_term_counts.pop()
                    if line_count:
                        line_count.pop()
                    if len(preamble_formatting) > 1:
                        line_formatting = preamble_formatting.pop()
                    else:
                        preamble_formatting.pop()
        # If leftover counts, add them
        while preamble_formatting:
            if not line_count:
                line_count.append(1)
            total_count += line_count[-1] * preamble_term_counts[-1]
            preamble_formatting.pop()
            preamble_term_counts.pop()
            line_count.pop()
        if total_label:
            return (
                doc.index
                + tuple(total_count)
                + (sum(total_count),)
            )
        return (
            doc.index
            + tuple(total_count)
        )


commands['enhanced_count_occurrences'] = EnhancedOccurrenceCounter


class SubjectCounter():

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
        return ('term', 'subject', 'count',)

    @staticmethod
    @check_spacy
    def process_document(doc, terms, pattern):
        terms_sorted = sorted(terms, key=len, reverse=True)
        term_pattern = re.compile(pattern.format('|'.join(terms_sorted)))

        def clean_text(text):
            '''Text cleaner specifically for this analysis.'''
            # remove bullet points from beginning of lines
            text = re.sub(r'\n\( ?\w+ ?\)', '', text)
            # convert parentheses into commas
            text = re.sub(r'(?: \(|\) )', ', ', text)
            # convert colons into periods (only if followed by a new sentence)
            text = re.sub(r'\: (?=[A-Z])', '. ', text)
            # remove odd characters
            text = re.sub(r'[^a-zA-Z\d\'\.\!\?\:\;\,\s]', ' ', text)
            # remove part numbers (e.g. 50.11) from beginning of lines
            text = re.sub(r'\n(\d+\.\d+\w? ?)+', '', text)
            # remove "that" which causes parsing problems
            text = text.replace('that', '')
            # remove extraneous whitespaces
            text = text.replace('\n\r', ' ')
            return re.sub(r'\s+', ' ', text).strip()

        try:
            spacy_text = nlp(clean_text(doc.text))
        except ValueError:
            print(f'Document {doc.index} is too large for spacy. '
                   'Split into multiple documents and try again.')
            return
        subject_terms = []
        for sentence in spacy_text.sents:
            for token in sentence:
                if term_pattern.search(token.text):
                    subject = ''
                    term = token.text
                    try:
                        verb = [a for a in token.ancestors if a.dep_ == 'ROOT'][0]
                    except IndexError:
                        verb = token
                    # if ROOT of sentence is not a verb, make term root verb
                    if verb.pos_ != 'VERB':
                        verb = token
                    # if root verb is "apply to," find the noun-object of the sentence
                    if verb.text == 'apply':
                        try:
                            prep = [c for c in verb.children if c.text == 'to'][0]
                            subject = [c for c in prep.children if 'obj' in c.dep_]
                        except IndexError:
                            pass
                    # find noun-subject for root verb
                    if not subject:
                        subject = [c for c in verb.children if 'subj' in c.dep_]
                    # if no noun-subject, look for a "it" or "there"
                    if not subject:
                        subject = [c for c in verb.children if 'expl' in c.dep_]
                    # if subject a stop word ("he" or "it") and sentence begins with "if",
                    # look for noun-subject at beginning of sentence
                    if subject:
                        if subject[0].is_stop and sentence.text.lower().startswith('if'):
                            subject = [t for t in sentence if 'nsubj' in t.dep_]
                    # finally, look for a noun-subject anywhere in the sentence
                    if not subject:
                        subject = [t for t in sentence if 'nsubj' in t.dep_]
                    # if no subject found, return empty string
                    if not subject:
                        subject_terms.append(doc.index + (term, ''))
                        continue
                    subject = subject[0].text.strip().lower()
                    subject_terms.append(doc.index + (term, subject))
        return list(k + (v,) for k, v in collections.Counter(subject_terms).items())


commands['count_subjects'] = SubjectCounter


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
