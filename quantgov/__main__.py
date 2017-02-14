"""
Quantgov: a policy analytics framework 
"""
import argparse
import io
import logging
import shutil
import zipfile

import requests

from pathlib import Path

log = logging.getLogger(__name__)


_URL = 'https://github.com/QuantGov/{component}/archive/{parent}.zip'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command')
    create = subparsers.add_parser('start')
    create.add_argument(
        'component', choices=['corpus', 'estimator', 'project'])
    create.add_argument('path', type=Path)
    create.add_argument('--parent', default='master')
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


def main():
    args = parse_args()
    if args.command == 'start':
        start_component(args)


if __name__ == '__main__':
    main()
