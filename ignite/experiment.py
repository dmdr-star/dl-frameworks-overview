import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import resnet18

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, RunningAverage


def get_loaders():
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
    return train_loader, valid_loader


def get_model():
    return resnet18(num_classes=10)


def run(epochs, lr, display_gpu_info):
    train_loader, val_loader = get_loaders()
    model = get_model()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)  # Move model before creating optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "nll": Loss(F.cross_entropy)}, device=device
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    if display_gpu_info:
        from ignite.contrib.metrics import GpuInfo

        GpuInfo().attach(trainer, name="gpu")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    run(epochs=3, lr=1e-3, display_gpu_info=True)
