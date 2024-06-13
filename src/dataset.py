import lightning as L
import numpy as np
import polars as pl
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import config
from preprocess import normalize

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


class LMDataset(Dataset):
    def __init__(self, df, tokenizer, stage="train"):
        assert stage in ["train", "val", "test"]
        self.tokenizer = tokenizer
        self.stage = stage
        df = (
            HFDataset.from_pandas(df.to_pandas())
            .map(self.tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            .to_pandas()
        )
        self.df = pl.from_pandas(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self._generate_data(index)
        data["label"] = self._generate_label(index)
        return data

    def _generate_data(self, index):
        data = {
            "input_ids": np.array(self.df[index, "input_ids"]),
            "attention_mask": np.array(self.df[index, "attention_mask"]),
        }
        return data

    def _generate_label(self, index):
        if self.stage == "test":
            return np.array([0, 0, 0])
        else:
            return self.df[index, config.PROTEIN_NAMES].to_numpy()[0]

    def tokenize(self, batch, tokenizer):
        output = tokenizer(batch["molecule_smiles"], truncation=True)
        return output


class LBDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer

    # def prepare_data_per_node(self, *args, **kwargs):
    #     return super().prepare_data_per_node(*args, **kwargs)

    # def prepare_data(self) -> None:
    #     return super().prepare_data()

    # def setup(self, stage=None):
    #     return super().setup(stage)

    def _generate_dataset(self, stage):
        if stage == "train":
            df = self.train_df
        elif stage == "val":
            df = self.val_df
        elif stage == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        dataset = LMDataset(df, self.tokenizer, stage=stage)
        return dataset

    def _generate_dataloader(self, stage):
        dataset = self._generate_dataset(stage)
        if stage == "train":
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        return DataLoader(
            dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )

    def train_dataloader(self):
        return self._generate_dataloader("train")

    def val_dataloader(self):
        return self._generate_dataloader("val")

    def test_dataloader(self):
        return self._generate_dataloader("test")
