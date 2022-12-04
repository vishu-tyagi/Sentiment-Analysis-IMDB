import os
import gc
import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from sentiment_analysis.config import SentimentAnalysisConfig
from sentiment_analysis.features.helpers import tfidf_init, stopwords_init
from sentiment_analysis.utils import timing
from sentiment_analysis.utils.constants import (
    DATA_DIR,
    TEXT,
    TARGET,
    ORIGINAL_TEXT,
    SPLIT,
    DEVELOP,
    TRAIN,
    VALID,
    TEST
)

logger = logging.getLogger(__name__)

class Features():
    def __init__(self, config: SentimentAnalysisConfig = SentimentAnalysisConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))

        self.vectorizer = tfidf_init()
        self.stopwords = stopwords_init()
        self.lemmatizer = WordNetLemmatizer()

    @timing
    def build(self, df: pd.DataFrame):
        df[ORIGINAL_TEXT] = df[TEXT]
        df[TARGET] = df[TARGET].map(self.config.SENTIMENT_MAP)
        df = self.clean(df)
        dev = df[df[SPLIT].isin([DEVELOP])].copy()
        test = df[df[SPLIT].isin([TEST])].copy()
        train, valid, _, _ = train_test_split(
            dev, dev[TARGET], test_size=.1, shuffle=True, random_state=64
        )
        train[SPLIT], valid[SPLIT] = TRAIN, VALID
        train = self.fit_transform(train)
        valid, test = self.transform(valid), self.transform(test)
        df = pd.concat([train, valid, test], ignore_index=True).copy()
        return df

    @timing
    def fit(self, df: pd.DataFrame):
        self.vectorizer.fit(df[TEXT].values)
        return self

    @timing
    def transform(self, df: pd.DataFrame):
        X = self.vectorizer.transform(df[TEXT].values)
        X = pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), X], axis=1).copy()
        return df

    @timing
    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    @timing
    def clean(self, df: pd.DataFrame):
        df = df.copy()
        # Convert to lower
        df[TEXT] = df[TEXT].str.lower()
        # Remove URLs
        df[TEXT] = df[TEXT].str.replace(r"http\S+|www.\S+", "", regex=True)
        # Remove HTML
        df[TEXT] = df[TEXT].str.replace("<[^<]+?>", " ", regex=True)
        # Remove symbols
        df[TEXT] = df[TEXT].str.replace(r"[/(){}\[\]\|@,;\-]", " ", regex=True)
        # Remove punctuation
        df[TEXT] = df[TEXT].str.replace(r"[^\w\s]", "", regex=True)
        # Tokenize
        df[TEXT] = df[TEXT].apply(lambda x: word_tokenize(x))
        # Remove stopwords
        df[TEXT] = df[TEXT].apply(lambda x: [word for word in x if word not in self.stopwords])
        # Lemmatize
        df[TEXT] = df[TEXT].apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])
        # Join and return
        df[TEXT] = df[TEXT].apply(lambda x: " ".join(x))
        return df
