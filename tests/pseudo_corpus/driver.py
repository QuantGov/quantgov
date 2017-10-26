import quantgov

from pathlib import Path

driver = quantgov.corpora.RecursiveDirectoryCorpusDriver(
    directory=Path(__file__).parent.joinpath('data', 'clean'),
    index_labels=('file',)
)
