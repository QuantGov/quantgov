import warnings

from ..corpus import (
    Document,
    CorpusStreamer,
    CorpusDriver,
    FlatFileCorpusDriver,
    RecursiveDirectoryCorpusDriver,
    NamePatternCorpusDriver,
    IndexDriver,
    S3Driver,
    S3DatabaseDriver
)

warnings.warn(
    ("quantgov.corpora has been moved to quantgov.corpus and will be removed"
     " in a future version."),
    DeprecationWarning)
