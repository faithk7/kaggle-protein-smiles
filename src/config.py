from pathlib import Path


class CFG:
    DATA_ROOT = Path("../data/")
    MODEL_ROOT = Path("../models/")
    TRAIN_PATH = DATA_ROOT / "train.csv"
    TEST_PATH = DATA_ROOT / "test.csv"

    DEBUG = True
    ENABLE_NEPTUNE = False
