import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from metrics import MAPMetric


class LMModel(nn.Module):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=3)
        self.lm = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, batch):
        last_hidden_state = self.lm(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        logits = self.classifier(self.dropout(last_hidden_state[:, 0]))
        return {
            "logits": logits,
        }

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"].float())
        output["loss"] = loss
        return output


class LBModelModule(L.LightningModule):
    def __init__(self, MODEL_NAME):
        super().__init__()
        self.model = LMModel(MODEL_NAME)
        self.map_ = MAPMetric(task="binary")

    def forward(self, batch):
        return self.model(batch)

    def calculate_loss(self, batch, batch_idx):
        return self.model.calculate_loss(batch)

    def training_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        self.log(
            "train_loss",
            ret["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return ret["loss"]

    def validation_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        self.log(
            "val_loss",
            ret["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.map_.update(F.sigmoid(ret["logits"]), batch["labels"].long())

    def on_validation_epoch_end(self):
        val_map = self.map_.compute()
        self.log(
            "val_map",
            val_map,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.map_.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward(batch)["logits"]
        probs = F.sigmoid(logits)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return {
            "optimizer": optimizer,
        }
