import logging

from sentiment_analysis.config import SentimentAnalysisConfig
from sentiment_analysis.data_access import DataClass
# from sentiment_analysis.features import Features
from sentiment_analysis.utils import timing

logger = logging.getLogger(__name__)


@timing
def fetch(config: SentimentAnalysisConfig = SentimentAnalysisConfig) -> None:
    logger.info("Fetching data...")
    data = DataClass(config)
    data.make_dirs()
    data.fetch()
    return
