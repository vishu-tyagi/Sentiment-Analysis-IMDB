import os
import logging
from pathlib import Path

import pandas as pd

from sentiment_analysis.config import SentimentAnalysisConfig
from sentiment_analysis.data_access.helpers import (
    download_data, unzip_tar, collect_files
)
from sentiment_analysis.utils.constants import (
    DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
    TAR_FOLDER,
    TAR_RELEVANT_FOLDERS,
    TAR_RELEVANT_FILES_PATTERN,
    TEXT,
    TARGET,
    SPLIT,
    DEVELOP,
    TRAIN,
)

logger = logging.getLogger(__name__)


class DataClass():
    def __init__(
        self, config: SentimentAnalysisConfig = SentimentAnalysisConfig
    ) -> None:
        self.config = config
        self.data_url = config.DATA_URL
        self.sentiment_map = config.SENTIMENT_MAP

        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))
        self.model_path = Path(os.path.join(self.current_path, MODEL_DIR))
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

    def make_dirs(self):
        dirs = [
            self.data_path,
            self.model_path,
            self.reports_path
        ]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory {self.data_path}")
        logger.info(f"Created model directory {self.model_path}")
        logger.info(f"Created reports directory {self.reports_path}")

    def fetch(self):
        file_name = self.data_url.split('/')[-1]
        file_path = Path(os.path.join(self.data_path, file_name))
        logger.info(f"Fetching raw data ...")
        download_data(url=self.data_url, save_to=file_path)
        logger.info(f"Downloaded {file_name} to {self.data_path}")
        logger.info(f"Unpacking {file_name} ...")
        relevant_folders = \
            [os.path.join(TAR_FOLDER, f) for f in TAR_RELEVANT_FOLDERS]
        unzip_tar(file_path, relevant_folders, self.data_path)

    def build(self):
        dirs = [os.path.join(self.data_path, TAR_FOLDER, f) for f in TAR_RELEVANT_FOLDERS]
        files = collect_files(dirs=dirs, file_types=TAR_RELEVANT_FILES_PATTERN)
        review, sentiment, split = [], [], []
        for f in files:
            with open(f, "r") as r:
                review.append(r.read())
                sentiment.append(f.split('/')[-2])
                split.append(f.split('/')[-3])
        df = pd.DataFrame({TEXT: review, TARGET: sentiment, SPLIT: split})
        df[SPLIT] = df[SPLIT].replace({TRAIN: DEVELOP})
        return df