"""
Quantgov: a policy analytics framework
"""

import argparse
import csv
import io
import functools
import logging
import shutil
import sys
import zipfile

import requests

import joblib as jl
import quantgov

from pathlib import Path

log = logging.getLogger(__name__)


_URL = 'https://github.com/QuantGov/{component}/archive/{parent}.zip'
ENCODE_OUT = 'utf-8'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command')

    # Create command
    create = subparsers.add_parser('start')
    create.add_argument(
        'component', choices=['corpus', 'estimator', 'project'])
    create.add_argument('path', type=Path)
    create.add_argument('--parent', default='master')

    # NLP command
    nlp_subparser = subparsers.add_parser('nlp')
    nlp_subcommands = nlp_subparser.add_subparsers(dest='subcommand')
    for command, builtin in quantgov.nlp.commands.items():
        subcommand = nlp_subcommands.add_parser(
            command, help=builtin.cli.help)
        subcommand.add_argument(
            'corpus', help='Path to a QuantGov Corpus directory')
        for argument in builtin.cli.arguments:
            flags = ((argument.flags,) if isinstance(argument.flags, str)
                     else argument.flags)
            kwargs = {} if argument.kwargs is None else argument.kwargs
            subcommand.add_argument(*flags, **kwargs)
        subcommand.add_argument(
            '-o', '--outfile',
            type=lambda x: open(x, 'w', newline='', encoding=ENCODE_OUT),
            default=sys.stdout
        )

    # ML Command
    ml_parser = subparsers.add_parser('ml')
    ml_subcommands = ml_parser.add_subparsers(dest='subcommand')

    # ML Evaluate
    evaluate = ml_subcommands.add_parser(
        'evaluate', help='Evaluate candidate models')
    evaluate.add_argument(
        'modeldefs', type=Path,
        help='python module containing candidate models'
    )
    evaluate.add_argument(
        'trainers',
        type=quantgov.ml.Trainers.load,
        help='saved Trainers object'
    )
    evaluate.add_argument(
        'labels', type=quantgov.ml.Labels.load, help='saved Labels object')
    evaluate.add_argument(
        'output_results',
        type=lambda x: open(x, 'w', encoding=ENCODE_OUT),
        help='Output file for evaluation results'
    )
    evaluate.add_argument(
        'output_suggestion',
        type=lambda x: open(x, 'w', encoding=ENCODE_OUT),
        help='Output file for model suggestion'
    )
    evaluate.add_argument(
        '--folds', type=int, default=5,
        help='Number of folds for cross-validation')
    evaluate.add_argument('--scoring', default='f1', help='scoring method')

    # ML Train
    train = ml_subcommands.add_parser('train', help='Train a model')
    train.add_argument(
        'modeldefs', type=Path,
        help='Python module containing candidate models'
    )
    train.add_argument('configfile', help='Model configuration file')
    train.add_argument(
        'vectorizer',
        type=jl.load,
        help='saved Vectorizer object'
    )
    train.add_argument(
        'trainers',
        type=quantgov.ml.Trainers.load,
        help='saved Trainers object'
    )
    train.add_argument(
        'labels', type=quantgov.ml.Labels.load, help='saved Labels object')
    train.add_argument(
        '-o', '--outfile', help='location to save the trained Estimator'
    )

    # ML Estimate
    estimate = ml_subcommands.add_parser(
        'estimate', help='Estimate label values for a target corpus')
    estimate.add_argument(
        'estimator',
        type=quantgov.ml.Estimator.load,
        help='saved Estimator object'
    )
    estimate.add_argument(
        'corpus', type=quantgov.load_driver,
        help='Path to a QuantGov corpus')
    estimate.add_argument(
        '--probability', action='store_true',
        help='output probabilities instead of predictions')
    estimate.add_argument(
        '--precision', default=4, type=int,
        help='number of decimal places to round the probabilities')
    estimate.add_argument(
        '-o', '--outfile',
        type=lambda x: open(x, 'w', newline='', encoding='utf-8'),
        default=sys.stdout,
        help='location to save estimation results'
    )

    return parser.parse_args()


def download(component, parent, outdir):
    response = requests.get(
        _URL.format(component=component, parent=parent),
    )
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    for name in archive.namelist():
        if name.split('/', 1)[-1] == '':
            continue
        outfile = outdir.joinpath(name.split('/', 1)[-1])
        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True)
        if name.endswith('/'):
            outfile.mkdir()
            continue
        with outfile.open('wb') as outf, archive.open(name) as inf:
            outf.write(inf.read())


def start_component(args):
    if args.path.exists():
        log.error("A file or folder with that name already exists")
        exit(1)
    args.path.mkdir()
    try:
        download(args.component, args.parent, args.path)
    except Exception:
        shutil.rmtree(str(args.path))
        raise


def run_corpus_builtin(args):
    driver = quantgov.load_driver(args.corpus)
    writer = csv.writer(args.outfile)
    builtin = quantgov.nlp.commands[args.subcommand]
    func_args = {i: j for i, j in vars(args).items()
                 if i not in {'command', 'subcommand', 'outfile', 'corpus'}}
    writer.writerow(driver.index_labels + builtin.get_columns(func_args))
    partial = functools.partial(
        builtin.process_document,
        **func_args
    )
    for i in quantgov.utils.lazy_parallel(partial, driver.stream()):
        writer.writerow(i)
        args.outfile.flush()


def run_estimator(args):
    if args.subcommand == "evaluate":
        quantgov.ml.evaluate(
            args.modeldefs, args.trainers, args.labels, args.folds,
            args.scoring, args.output_results, args.output_suggestion
        )
    elif args.subcommand == "train":
        quantgov.ml.train_and_save_model(
            args.modeldefs, args.configfile, args.vectorizer, args.trainers,
            args.labels, args.outfile)
    elif args.subcommand == "estimate":
        writer = csv.writer(args.outfile)
        labels = args.corpus.index_labels
        if args.probability:
            if args.estimator.multilabel:
                if args.estimator.multiclass:
                    writer.writerow(labels + ('label', 'class', 'probability'))
                else:
                    writer.writerow(labels + ('label', 'probability'))
            elif args.estimator.multiclass:
                writer.writerow(labels + ('class', 'probability'))
            else:
                writer.writerow(
                    labels + ('{}_prob'.format(args.estimator.label_names[0]),)
                )
        else:
            if args.estimator.multilabel:
                writer.writerow(labels + ('label', 'prediction'))
            else:
                writer.writerow(
                    labels + ('{}'.format(args.estimator.label_names[0]),)
                )
        writer.writerows(
            docidx + result for docidx,
            result in quantgov.ml.estimate(
                args.estimator,
                args.corpus,
                args.probability,
                args.precision)
        )


def main():
    args = parse_args()
    {
        'start': start_component,
        'nlp': run_corpus_builtin,
        'ml': run_estimator,
    }[args.command](args)


if __name__ == '__main__':
    main()
