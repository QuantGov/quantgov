"""
quantgov.estimator.estimation

Functionality for making predictions with an estimator
"""
import csv
import logging

import sklearn.pipeline

log = logging.getLogger(__name__)


def get_pipeline(vectorizer, model):
    """
    Get the full estimation pipeline

    Arguments:
        * vectorizer: a sklearn Vectorizer (or pipeline)
        * model: a quantgov.estimator.Estimator

    Returns: a sklearn Pipeline
    """
    return sklearn.pipeline.Pipeline((
        ('vectorizer', vectorizer),
        ('model', model.model)
    ))


def estimate_simple(vectorizer, model, streamer):
    """
    Generate predictions for an estimator

    Arguments:
        * vectorizer: a sklearn Vectorizer (or pipeline)
        * model: a quantgov.estimator.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, prediction

    """
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    yield from zip(streamer.index, pipeline.predict(texts))


def estimate_probability(vectorizer, model, streamer):
    """
    Generate probabilities for a one-label estimator

    Arguments:
        * vectorizer: a sklearn Vectorizer (or pipeline)
        * model: a quantgov.estimator.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, probability

    """
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    truecol = list(int(i) for i in model.model.classes_).index(1)
    predicted = (i[truecol] for i in pipeline.predict_proba(texts))
    yield from zip(streamer.index, predicted)


def estimate_probability_multilabel(vectorizer, model, streamer):
    """
    Generate probabilities for a multilabel binary estimator

    Arguments:
        * vectorizer: a sklearn Vectorizer (or pipeline)
        * model: a quantgov.estimator.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, probability

    """
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    try:
        truecols = tuple(
            list(int(i) for i in label_classes).index(1)
            for label_classes in model.model.classes_
        )
    except (AttributeError, TypeError):
        truecols = tuple(
            list(int(i) for i in label_classes).index(1)
            for label_classes in (
                est.classes_ for est in model.model.steps[-1][-1].estimators_
            )
        )
    predicted = pipeline.predict_proba(texts)
    try:
        for i, docidx in enumerate(streamer.index):
            yield docidx, tuple(
                label_predictions[i, truecols[j]]
                for j, label_predictions in enumerate(predicted))
    except IndexError:
        yield from zip(streamer.index, predicted)


def estimate_probability_multiclass(vectorizer, model, streamer):
    """
    Generate probabilities for a one-label, multiclass estimator

    Arguments:
        * vectorizer: a sklearn Vectorizer (or pipeline)
        * model: a quantgov.estimator.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, probability

    """
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    yield from zip(streamer.index, pipeline.predict_proba(texts))


def estimate_probability_multilabel_multiclass(vectorizer, model, streamer):
    """
    Generate probabilities for a multilabel, multiclass estimator

    Arguments:
        * vectorizer: a sklearn Vectorizer (or pipeline)
        * model: a quantgov.estimator.Estimator
        * streamer: a quantgov.corpora.CorpusStreamer

    Yields:
        2-tuples of docindex, probability

    """
    pipeline = get_pipeline(vectorizer, model)
    texts = (doc.text for doc in streamer)
    predicted = pipeline.predict_proba(texts)
    for i, docidx in enumerate(streamer.index):
        yield docidx, tuple(label_predictions[i]
                            for label_predictions in predicted)


def is_multiclass(classes):
    """
    Returns True if values in classes are anything but 1, 0, True, or False,
    otherwise returns False.
    """
    try:
        return len(set(int(i) for i in classes) - {0, 1}) != 0
    except ValueError:
        return True


def estimate(vectorizer, model, corpus, probability, outfile):
    """
    Estimate label values for documents in corpus

    Arguments:

        * **vectorizer**: joblib-saved vectorizer
        * **model**: saved `quantgov.estimator.Model` object
        * **corpus**: path to a quantgov corpus
        * **probability**: if True, predict probability
        * **outfile**: open file object for writing results
    """
    streamer = corpus.get_streamer()
    writer = csv.writer(outfile)
    if len(model.label_names) > 1:
        multilabel = True
        try:
            multiclass = any(is_multiclass(i) for i in model.model.classes_)
        except (AttributeError, TypeError):
            multiclass = any(
                is_multiclass(i.classes_) for i in
                model.model.steps[-1][-1].estimators_
            )
    else:
        multilabel = False
        multiclass = is_multiclass(model.model.classes_)

    # TODO: This is very ugly and complicated and should probably be refactored
    if probability:
        if multilabel:
            if multiclass:  # Multilabel-multiclass probability
                results = estimate_probability_multilabel_multiclass(
                    vectorizer, model, streamer)
                writer.writerow(corpus.index_labels +
                                ('label', 'class', 'probability'))
                writer.writerows(
                    docidx + (label_name, class_name, prediction)
                    for docidx, predictions in results
                    for label_name, label_classes, label_predictions
                    in zip(
                        model.label_names, model.model.classes_, predictions)
                    for class_name, prediction
                    in zip(label_classes, label_predictions)
                )
            else:  # Multilabel probability
                results = estimate_probability_multilabel(
                    vectorizer, model, streamer)
                writer.writerow(corpus.index_labels + ('label', 'probability'))
                writer.writerows(
                    docidx + (label_name, prediction)
                    for docidx, predictions in results
                    for label_name, prediction
                    in zip(model.label_names, predictions)
                )
        elif multiclass:  # Multiclass probability
            writer.writerow(corpus.index_labels + ('class', 'probability'))
            results = estimate_probability_multiclass(
                vectorizer, model, streamer)
            writer.writerows(
                docidx + (class_name, prediction)
                for docidx, predictions in results
                for class_name, prediction in zip(
                    model.model.classes_, predictions)
            )
        else:  # Simple probability
            results = estimate_probability(vectorizer, model, streamer)
            writer.writerow(
                corpus.index_labels + (model.label_names[0] + '_prob',))
            writer.writerows(
                docidx + (prediction,) for docidx, prediction in results)
    elif multilabel:  # Multilabel Prediction
        results = estimate_simple(vectorizer, model, streamer)
        writer.writerow(corpus.index_labels + ('label', 'prediction'))
        writer.writerows(
            docidx + (label_name, prediction,)
            for docidx, predictions in results
            for label_name, prediction in zip(model.label_names, predictions)
        )
    else:  # Simple Prediction
        results = estimate_simple(vectorizer, model, streamer)
        writer.writerow(corpus.index_labels + model.label_names)
        writer.writerows(docidx + (prediction,)
                         for docidx, prediction in results)
