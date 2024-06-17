from pathlib import Path

import lightning as L
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from transformers import AutoTokenizer

from config import config
from dataset import LBDataModule, tokenizer
from model import LBModelModule
from preprocess import normalize
from utils import append_curtimestr, extract_modelname, parse_args, parse_yaml, timeit


@timeit
def main():
    df = pl.read_parquet(
        config.DATA_ROOT / "train.parquet",
        columns=["molecule_smiles", "protein_name", "binds"],
        n_rows=config.N_ROWS,
    )
    test_df = pl.read_parquet(
        config.DATA_ROOT / "test.parquet",
        columns=["molecule_smiles"],
        n_rows=10000 if config.DEBUG else None,
    )

    #! Refactor
    dfs = []
    for i, protein_name in enumerate(config.PROTEIN_NAMES):
        sub_df = df[i::3]
        sub_df = sub_df.rename({"binds": protein_name})
        if i == 0:
            dfs.append(sub_df.drop(["id", "protein_name"]))
        else:
            dfs.append(sub_df[[protein_name]])
        print(sub_df.columns)
        print(sub_df.shape)
        print(sub_df.head())
    df = pl.concat(dfs, how="horizontal")
    print(config.N_SAMPLES)
    df = df.sample(n=config.N_SAMPLES)
    print(df.shape)
    print(df.head())

    if config.NORMALIZE:
        df = df.with_columns(
            pl.col("molecule_smiles").map_elements(normalize, return_dtype=pl.Utf8)
        )
        test_df = test_df.with_columns(
            pl.col("molecule_smiles").map_elements(normalize, return_dtype=pl.Utf8)
        )

    #! Refactor
    train_df, val_df = df[: int(len(df) * 0.8)], df[int(len(df) * 0.8) :]

    datamodule = LBDataModule(train_df, val_df, test_df, tokenizer)
    modelmodule = LBModelModule(config.MODEL_NAME)

    save_path = config.MODEL_ROOT / append_curtimestr(
        extract_modelname(config.MODEL_NAME)
    )

    checkpoint_callback = ModelCheckpoint(
        filename="model-{val_map:.4f}",
        save_weights_only=True,
        monitor="val_map",
        mode="max",
        dirpath=save_path,
        save_top_k=1,
        verbose=1,
    )
    early_stop_callback = EarlyStopping(monitor="val_map", mode="max", patience=3)
    progress_bar_callback = TQDMProgressBar(refresh_rate=1)
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        progress_bar_callback,
    ]

    trainer = L.Trainer(callbacks=callbacks, **config.TRAINER_PARAMS)
    trainer.fit(modelmodule, datamodule)

    if config.DO_INFERENCE:
        do_inference(save_path, datamodule, trainer)


def eval_config() -> None:
    args = parse_args()

    # check if YAML path is provided, if so, load the config from the YAML file
    if args.config_path:
        parse_yaml(args.config_path)

    if config.DEBUG:
        config.N_SAMPLES = 1000
        config.TRAINER_PARAMS["max_epochs"] = 1


def do_inference(model_dir: Path, datamodule: LBDataModule, trainer):
    # get the test data
    test_df = pl.read_parquet(
        config.DATA_ROOT / "test.parquet",
        columns=["molecule_smiles"],
        n_rows=10000 if config.DEBUG else None,
    )
    model_paths = model_dir.glob("*.ckpt")
    test_dataloader = datamodule.test_dataloader()

    for model_path in model_paths:
        _do_inference_one_model(model_path, test_df, test_dataloader, trainer)

    _ensemble_submissions(model_dir)


#! refactor
def _do_inference_one_model(model_path: Path, test_df, test_dataloader, trainer):
    model = LBModelModule.load_from_checkpoint(model_path, model_name=config.MODEL_NAME)
    predictions = trainer.predict(model, test_dataloader)
    predictions = torch.cat(predictions).numpy()
    model_dir = model_path.parent

    pred_dfs = []
    for i, protein_name in enumerate(config.PROTEIN_NAMES):
        pred_dfs.append(
            test_df.with_columns(
                pl.lit(protein_name).alias("protein_name"),
                pl.lit(predictions[:, i]).alias("binds"),
            )
        )
    pred_df = pl.concat(pred_dfs)

    submit_df = (
        pl.read_parquet(
            Path(config.DATA_ROOT, "test.parquet"),
            columns=["id", "molecule_smiles", "protein_name"],
        )
        .join(pred_df, on=["molecule_smiles", "protein_name"], how="left")
        .select(["id", "binds"])
        .sort("id")
    )
    submit_df.write_csv(Path(model_dir, f"submission_{model_path.stem}.csv"))


def _ensemble_submissions(model_dir: Path):
    submission_paths = model_dir.glob("submission_*.csv")
    sub_dfs = []
    for submission_path in submission_paths:
        sub_dfs.append(pl.read_csv(submission_path))
    submit_df = pl.concat(sub_dfs).group_by("id").agg(pl.col("binds").mean()).sort("id")
    submit_df.write_csv(Path(model_dir, "submission.csv"))


if __name__ == "__main__":
    eval_config()
    main()
