import catalyst.dl as dl

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class MyConfigRunner(dl.SupervisedConfigRunner):
    def get_datasets(self, stage: str) -> "OrderedDict[str, Dataset]":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = CIFAR10(root="./data/train", train=True, download=True, transform=transform)
        valid_dataset = CIFAR10(root="./data/valid", train=False, download=True, transform=transform)

        return {"train": train_dataset, "valid": valid_dataset}

    def handle_batch(self, batch):
        x, y = batch["features"], batch["targets"]
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}
