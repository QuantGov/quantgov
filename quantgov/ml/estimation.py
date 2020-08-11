"""
quantgov.ml.estimation

Functionality for making predictions with an estimator
"""
import logging

log = logging.getLogger(__name__)


def estimate_simple(estimator, streamer):
    """
    Generate predictions for a one-label estimator

    Arguments:
        * estimator: a quantgov.ml.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, (prediction,)

    """
    texts = (doc.text for doc in streamer)
    predicted = estimator.pipeline.predict(texts)
    for docidx, prediction in zip(streamer.index, predicted):
        yield docidx, (prediction,)


def estimate_multilabel(estimator, streamer):
    """
    Generate predictions for a multi-label estimator

    Arguments:
        * estimator: a quantgov.ml.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, (label, prediction,)

    """
    for docidx, (prediction,) in estimate_simple(estimator, streamer):
        for label, label_prediction in zip(estimator.label_names, prediction):
            yield docidx, (label, label_prediction)


def estimate_probability(estimator, streamer, precision):
    """
    Generate probabilities for a one-label estimator

    Arguments:
        * estimator: a quantgov.ml.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, (probability,)

    """
    texts = (doc.text for doc in streamer)
    truecol = list(int(i) for i in estimator.pipeline.classes_).index(1)
    predicted = (
        estimator.pipeline.predict_proba(texts)[:, truecol].round(precision))
    yield from zip(streamer.index, ((prob,) for prob in predicted))


def estimate_probability_multilabel(estimator, streamer, precision):
    """
    Generate probabilities for a multilabel binary estimator

    Arguments:
        * estimator: a quantgov.ml.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, (label, probability)

    """
    texts = (doc.text for doc in streamer)
    model = estimator.pipeline.steps[-1][1]
    try:
        truecols = tuple(
            list(int(i) for i in label_classes).index(1)
            for label_classes in model.classes_
        )
    except (AttributeError, TypeError):
        truecols = tuple(
            list(int(i) for i in label_classes).index(1)
            for label_classes in (
                est.classes_ for est in model.steps[-1][1].estimators_
            )
        )
    predicted = estimator.pipeline.predict_proba(texts).round(int(precision))

    try:
        yield from (
            (docidx, (label, label_prediction[truecol]))
            for docidx, doc_predictions in zip(streamer.index, predicted)
            for label, label_prediction, truecol
            in zip(estimator.label_names, doc_predictions, truecols)
        )
    except IndexError:
        yield from (
            (docidx, (label, label_prediction))
            for docidx, doc_predictions in zip(streamer.index, predicted)
            for (label, label_prediction)
            in zip(estimator.label_names, doc_predictions)
        )


def estimate_probability_multiclass(estimator, streamer, precision):
    """
    Generate probabilities for a one-label, multiclass estimator

    Arguments:
        * estimator: a quantgov.ml.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, (class, probability)

    """
    texts = (doc.text for doc in streamer)
    probs = estimator.pipeline.predict_proba(texts).round(precision)
    yield from (
        (docidx, (class_, probability))
        for docidx, doc_probs in zip(streamer.index, probs)
        for class_, probability in zip(estimator.pipeline.classes_, doc_probs)
    )


def estimate_probability_multilabel_multiclass(estimator, streamer, precision):
    """
    Generate probabilities for a multilabel, multiclass estimator

    Arguments:
        * estimator: a quantgov.ml.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, (label, class, probability

    """
    texts = (doc.text for doc in streamer)
    probs = estimator.pipeline.predict_proba(texts)
    yield from (
        (docidx, (label_name, class_, prob))
        for label_name, label_probs in zip(estimator.label_names, probs)
        for docidx, doc_probs in zip(streamer.index, label_probs)
        for class_, prob in zip(estimator.pipeline.classes_, doc_probs)
    )


def estimate(estimator, corpus, probability, precision=4, *args, **kwargs):
    """
    Estimate label values for documents in corpus

    Arguments:

        * **estimator**: path to a saved `quantgov.ml.Estimator` object
        * **corpus**: path to a quantgov corpus
        * **probability**: if True, predict probability
        * **precision**: precision for probability prediction
    """
    streamer = corpus.get_streamer(*args, **kwargs)
    if probability:
        if estimator.multilabel:
            if estimator.multiclass:  # Multilabel-multiclass probability
                yield from estimate_probability_multilabel_multiclass(
                    estimator, streamer, precision)
            else:  # Multilabel probability
                yield from estimate_probability_multilabel(
                    estimator, streamer, precision)
        elif estimator.multiclass:  # Multiclass probability
            yield from estimate_probability_multiclass(
                estimator, streamer, precision)
        else:  # Simple probability
            yield from estimate_probability(
                estimator, streamer, precision)
    elif estimator.multilabel:  # Multilabel Prediction
        yield from estimate_multilabel(estimator, streamer)
    else:  # Binary and Multiclass
        yield from estimate_simple(estimator, streamer)
