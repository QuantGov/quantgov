# TODO: Docstrings

import collections
import concurrent.futures
import importlib
import os
import sys

from pathlib import Path


def load_driver(driver_path):
    """
    Load quantgov driver from a local file

    Arguments:
    * driver_path: relative path to the driver file
    """
    corpus = Path(driver_path)
    if not corpus.is_dir():
        corpus = corpus.parent
    sys.path.insert(0, str(corpus))
    if Path(driver_path).is_dir():
        mod = importlib.import_module(str(corpus), 'driver')
    else:
        mod = importlib.import_module(Path(driver_path).stem)
    driver = getattr(mod, 'driver')
    sys.path.pop(0)
    return driver


_POOLS = {
    'thread': concurrent.futures.ThreadPoolExecutor,
    'process': concurrent.futures.ProcessPoolExecutor
}


def lazy_parallel(func, *iterables, **kwargs):
    """
    Parallel execution without fully loading iterables

    Arguments:
    * func: function to call
    * iterables: any number of iterables, which will be passed to func as
      arguments

    Keyword Arugments:
    * max_workers: max number of threads or processes. Defaults to None.
    * worker: 'thread' (default) or 'process'
    """
    worker = kwargs.get('worker', 'thread')
    max_workers = kwargs.get('max_workers')
    if max_workers is None:  # Not in back-port
        max_workers = (os.cpu_count() or 1)
        if worker == 'thread':
            max_workers *= 5
    try:
        pooltype = _POOLS[worker]
    except KeyError:
        raise ValueError("Valid choices for worker are: {}"
                         .format(', '.join(_POOLS.keys())))
    jobs = []
    argsets = zip(*iterables)
    with pooltype(max_workers) as pool:
        for argset in argsets:
            jobs.append(pool.submit(func, *argset))
            if len(jobs) == pool._max_workers:
                yield jobs.pop(0).result()
        for job in jobs:
            yield job.result()


CLISpec = collections.namedtuple('CLISpec', ['help', 'arguments'])
CLIArg = collections.namedtuple('CLIArg', ['flags', 'kwargs'])
