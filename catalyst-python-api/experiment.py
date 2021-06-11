import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import resnet18

import catalyst.dl as dl


class MyRunner(dl.IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return dl.DeviceEngine(self._device)

    def get_callbacks(self, stage: str):
        return {
            "progress": dl.TqdmCallback(),
            "timer": dl.TimerCallback(),
            "criterion": dl.CriterionCallback(metric_key="loss", input_key="logits", target_key="targets"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "accuracy": dl.AccuracyCallback(input_key="logits", target_key="targets"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="accuracy",
                minimize=False,
                save_n_best=3,
                load_on_stage_start="best",
            ),
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["stage_0", "stage_1"]

    def get_stage_len(self, stage: str) -> int:
        return 3

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        BATCH_SIZE = 128
        NUM_WORKERS = 32
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = CIFAR10(root="./data/train", train=True, download=True, transform=transform)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS
        )

        valid_dataset = CIFAR10(root="./data/valid", train=False, download=True, transform=transform)
        valid_loader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS
        )

        return {"train": train_loader, "valid": valid_loader}

    def get_model(self, stage: str):
        return resnet18(num_classes=10)

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model, stage: str):
        return optim.Adam(model.parameters(), lr=1e-3)

    def get_scheduler(self, optimizer, stage: str):
        return None

    def get_trial(self):
        return None

    def get_loggers(self):
        return {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


if __name__ == "__main__":
    runner = MyRunner("./logs/catalyst-python-api", "cuda:0")
    runner.run()
