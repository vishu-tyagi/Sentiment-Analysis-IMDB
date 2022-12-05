from typing import (List, Optional)


class SentimentAnalysisConfig():
    # URL for downloading data
    DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    CURRENT_PATH = None

    SENTIMENT_MAP = {"pos": 1, "neg": 0}

    STOPWORDS_TO_ADD: Optional[List[str]] = []
    STOPWORDS_TO_DELETE: Optional[List[str]] = []

    TFIDF_ANALYZERS = {"char", "word"}
    TFIDF_CHAR_PARAMETERS = {
        "analyzer": "char",
        "ngram_range": (3, 3),
        "max_features": 4000,
        "min_df": 0.001,
        "max_df": 0.75
    }
    TFIDF_WORD_PARAMETERS = {
        "analyzer": "word",
        "ngram_range": (2, 2),
        "max_features": 1000,
        "min_df": 0.001,
        "max_df": 0.75
    }

    XGB_NUM_BOOST_ROUND = 3000
    XGB_EARLY_STOPPING_ROUNDS = 150
    XGB_PARAMETERS = {
        "booster": "gbtree",
        "nthread": 1,
        "disable_default_eval_metric": 1,
        "eta": 0.01,
        "gamma": 2.0,
        "max_depth": 5,
        "min_child_weight": 1,
        "max_delta_step": 0.0,
        "subsample": 0.7,
        "sampling_method": "uniform",
        "lambda": 1.0,
        "alpha": 0.2,
        "tree_method": "auto",
        "grow_policy": "lossguide"
    }
    XGB_OBJECTIVE = "binary:logistic"
    XGB_EVALUATION_METRIC = ""
    XGB_EVAL_FBETA = 1