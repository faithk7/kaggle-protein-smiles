import os
from pathlib import Path
from tkinter import NORMAL


class config:
    DATA_ROOT = (
        Path("../data/")
        if not os.path.exists("/kaggle")
        else Path("/kaggle/input/leash-BELKA")
    )
    MODEL_ROOT = Path("../models/")
    TRAIN_PATH = DATA_ROOT / "train.csv"
    TEST_PATH = DATA_ROOT / "test.csv"

    DEBUG = True
    ENABLE_NEPTUNE = False

    # -- data --
    N_ROWS = 180_000_000
    N_SAMPLES = 10_000 if DEBUG else 2_000_000
    PROTEIN_NAMES = ["BRD4", "HSA", "sEH"]

    # -- train --
    TRAIN_BATCH_SIZE = 256

    #! todo: refactor
    TRAINER_PARAMS = {
        "max_epochs": 5,
        "enable_progress_bar": True,
        "accelerator": "auto",
        "precision": "16-mixed",
        "gradient_clip_val": None,
        "accumulate_grad_batches": 1,
        "devices": [0],
    }

    MODEL_NAME = "DeepChem/ChemBERTa-10M-MTR"

    NORMALIZE = True
