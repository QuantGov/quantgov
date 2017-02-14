# TODO: Docstrings
import concurrent.futures

_POOLS = {
    'thread': concurrent.futures.ThreadPoolExecutor,
    'process': concurrent.futures.ProcessPoolExecutor
}


def lazy_parallel(func, *iterables, max_workers=None, worker='thread'):
    # TODO #DOCSTRING
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
