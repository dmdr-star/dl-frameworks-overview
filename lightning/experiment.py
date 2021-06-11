from argparse import ArgumentParser

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import resnet18

import pytorch_lightning as pl


BATCH_SIZE = 128
NUM_WORKERS = 32


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(num_classes=10)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class CIFAR10DataModule(pl.LightningDataModule):
    def setup(self, stage):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_dataset = CIFAR10(root="./data/train", train=True, download=True, transform=transform)
        self.valid_dataset = CIFAR10(root="./data/valid", train=False, download=True, transform=transform)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=NUM_WORKERS
        )
        return valid_loader


def main(args):
    data_module = CIFAR10DataModule()
    model = CIFAR10Classifier()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
