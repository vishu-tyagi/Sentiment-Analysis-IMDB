import os
from pathlib import Path

import numpy as np


class SentimentAnalysisConfig():
    # URL for downloading data
    DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    CURRENT_PATH = None

    SENTIMENT_MAP = {"pos": 1, "neg": 0}
