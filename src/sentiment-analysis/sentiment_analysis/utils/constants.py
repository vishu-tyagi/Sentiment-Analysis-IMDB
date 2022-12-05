DATA_DIR = "data"
MODEL_DIR = "model"
REPORTS_DIR = "reports"
SAVED_MODELS = "saved_models"

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
TAR_FOLDER = "aclImdb"
TAR_RELEVANT_FOLDERS = [
    "train/neg/", "train/pos/",
    "test/neg/", "test/pos/"
]
TAR_RELEVANT_FILES_PATTERN = ["*.txt"]

TEXT = "review"
TARGET = "sentiment"
ORIGINAL_TEXT = "Original Text"
SPLIT = "Split"
DEVELOP = "development"
TRAIN = "train"
VALID = "validation"
TEST = "test"
PREDICTION = "prediction"

ACCURACY = "Accuracy"
PRECISION = "Precision"
RECALL = "Recall"
F1_SCORE = "F1 Score"
