from torchmetrics import AveragePrecision


class MAPMetric(AveragePrecision):
    def __init__(self, task="binary"):
        super().__init__(task=task)
