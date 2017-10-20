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

import quantgov
import quantgov.corpora.builtins

from pathlib import Path

log = logging.getLogger(__name__)


_URL = 'https://github.com/QuantGov/{component}/archive/{parent}.zip'
ENCODE_OUT = 'utf-8'


def get_public_functions(library):
    return [
        i for i in vars(library).values()
        if hasattr(i, 'cli')
        and isinstance(getattr(i, 'cli'), quantgov.utils.CLISpec)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command')
    create = subparsers.add_parser('start')
    create.add_argument(
        'component', choices=['corpus', 'estimator', 'project'])
    create.add_argument('path', type=Path)
    create.add_argument('--parent', default='master')
    corpus = subparsers.add_parser('corpus')
    corpus_subcommands = corpus.add_subparsers(dest='subcommand')
    for func in get_public_functions(quantgov.corpora.builtins):
        subcommand = corpus_subcommands.add_parser(
            func.__name__, help=func.cli.help)
        subcommand.add_argument(
            'corpus', help='Path to a QuantGov Corpus directory')
        for argument in func.cli.arguments:
            flags = ((argument.flags,) if isinstance(argument.flags, str)
                     else argument.flags)
            kwargs = {} if argument.kwargs is None else argument.kwargs
            subcommand.add_argument(*flags, **kwargs)
        subcommand.add_argument(
            '-o', '--outfile',
            type=lambda x: open(x, 'w', newline='', encoding=ENCODE_OUT),
            default=io.TextIOWrapper(
                sys.stdout.buffer, encoding=ENCODE_OUT))
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
    except:
        shutil.rmtree(str(args.path))
        raise


def run_corpus_builtin(args):
    driver = quantgov.load_driver(args.corpus)
    writer = csv.writer(args.outfile)
    basefunc = getattr(quantgov.corpora.builtins, args.subcommand)
    writer.writerow(driver.index_labels + basefunc.get_columns(args))
    partial = functools.partial(
        basefunc,
        **{i: j for i, j in vars(args).items()
           if i not in {'command', 'subcommand', 'outfile', 'corpus'}}
    )
    for i in quantgov.utils.lazy_parallel(partial, driver.stream()):
        writer.writerow(i)
        args.outfile.flush()


def main():
    args = parse_args()
    {
        'start': start_component,
        'corpus': run_corpus_builtin
    }[args.command](args)


if __name__ == '__main__':
    main()
