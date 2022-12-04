from typing import (List, Optional)


class SentimentAnalysisConfig():
    # URL for downloading data
    DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    CURRENT_PATH = None

    SENTIMENT_MAP = {"pos": 1, "neg": 0}

    STOPWORDS_TO_ADD: Optional[List[str]] = []
    STOPWORDS_TO_DELETE: Optional[List[str]] = []

    TFIDF_ANALYZERS = {"word"}
    TFIDF_CHAR_PARAMETERS = {
        "analyzer": "char",
        "ngram_range": (3, 3),
        "max_features": 4000,
        "min_df": 0.001,
        "max_df": 0.75
    }
    TFIDF_WORD_PARAMETERS = {
        "analyzer": "word",
        "ngram_range": (2, 3),
        "max_features": 5000,
        "min_df": 0.001,
        "max_df": .75
    }