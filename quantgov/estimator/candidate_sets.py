import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.multioutput
import sklearn.pipeline
import sklearn.feature_extraction

import quantgov.estimator

classification = [
    quantgov.estimator.CandidateModel(
        name="Random Forests",
        model=sklearn.pipeline.Pipeline(steps=(
            ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
            ('rf', sklearn.ensemble.RandomForestClassifier(n_jobs=-1)),
        )),
        parameters={
            'rf__n_estimators': [5, 10, 25, 50, 100],
        }
    ),
    quantgov.estimator.CandidateModel(
        name="Logistic Regression",
        model=sklearn.pipeline.Pipeline(steps=(
            ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
            ('logit', sklearn.linear_model.LogisticRegression()),
        )),
        parameters={
            'logit__C': np.logspace(-2, 2, 5)
        }
    ),
]


multilabel_classification = [
    quantgov.estimator.CandidateModel(
        name="Random Forests",
        model=sklearn.pipeline.Pipeline(steps=(
            ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
            ('rf', sklearn.ensemble.RandomForestClassifier(n_jobs=-1)),
        )),
        parameters={
            'rf__n_estimators': [5, 10, 25, 50, 100],
        }
    ),
    quantgov.estimator.CandidateModel(
        name="Logistic Regression",
        model=sklearn.pipeline.Pipeline(steps=(
            ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
            ('logit', sklearn.multioutput.MultiOutputClassifier(
                sklearn.linear_model.LogisticRegression(),
                n_jobs=-1
            )),
        )),
        parameters={
            'logit__estimator__C': np.logspace(-2, 2, 5)
        }
    ),
]
