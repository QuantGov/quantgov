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

import joblib as jl
import requests

import quantgov
import quantgov.corpora.builtins

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

    # Corpus command
    corpus = subparsers.add_parser('corpus')
    corpus_subcommands = corpus.add_subparsers(dest='subcommand')
    for command, builtin in quantgov.corpora.builtins.commands.items():
        subcommand = corpus_subcommands.add_parser(
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

    # Estimator Command
    estimator = subparsers.add_parser('estimator')
    estimator_subcommands = estimator.add_subparsers(dest='subcommand')

    # Estimator Evaluate
    evaluate = estimator_subcommands.add_parser(
        'evaluate', help='Evaluate candidate models')
    evaluate.add_argument(
        'modeldefs', type=Path,
        help='python module containing candidate models'
    )
    evaluate.add_argument(
        'trainers', type=jl.load, help='saved Trainers object')
    evaluate.add_argument(
        'labels', type=jl.load, help='saved Labels object')
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

    # Estimator Train
    train = estimator_subcommands.add_parser('train', help='Train a model')
    train.add_argument(
        'modeldefs', type=Path,
        help='Python module containing candidate models'
    )
    train.add_argument('configfile', help='Model configuration file')
    train.add_argument(
        'trainers', type=jl.load, help='saved Trainers object')
    train.add_argument(
        'labels', type=jl.load, help='saved Labels object')
    train.add_argument(
        '-o', '--outfile', help='location to save the trained model'
    )

    # Estimator Estimate
    estimate = estimator_subcommands.add_parser(
        'estimate', help='Estimate label values for a target corpus')
    estimate.add_argument(
        'vectorizer', type=jl.load,
        help='joblib-saved scikit-learn vectorizer'
    )
    estimate.add_argument(
        'model', type=jl.load,
        help='saved Model object'
    )
    estimate.add_argument(
        'corpus', type=quantgov.load_driver,
        help='Path to a QuantGov corpus')
    estimate.add_argument(
        '--probability', action='store_true',
        help='output probabilities instead of predictions')
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
    builtin = quantgov.corpora.builtins.commands[args.subcommand]
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
        quantgov.estimator.evaluate(
            args.modeldefs, args.trainers, args.labels, args.folds,
            args.scoring, args.output_results, args.output_suggestion
        )
    elif args.subcommand == "train":
        quantgov.estimator.train_and_save_model(
            args.modeldefs, args.configfile, args.trainers, args.labels,
            args.outfile)
    elif args.subcommand == "estimate":
        quantgov.estimator.estimate(
            args.vectorizer, args.model, args.corpus, args.probability,
            args.outfile
        )


def main():
    args = parse_args()
    {
        'start': start_component,
        'corpus': run_corpus_builtin,
        'estimator': run_estimator
    }[args.command](args)


if __name__ == '__main__':
    main()
