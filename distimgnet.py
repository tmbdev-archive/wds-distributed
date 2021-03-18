import os
import torch
import torch.distributed as dist
import webdataset as wds
import numpy as np
import time
import typer
from itertools import islice
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import models


def every(seconds, start=None):
    start = time.time() if start is None else start
    last = [start]

    def f():
        now = time.time()
        if now - last[0] > seconds:
            last[0] = now
            return True
        else:
            return False

    return f


class Trainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.last_lr = None
        self.device = "cpu"
        self.clip_grad = 1.0
        self.batches = 0
        self.samples = 0
        self.set_lr(0.1)
        self.steps = []
        self.losses = []

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.criterion.to(device)

    def set_last(self, *args):
        self.last = [x.detach().cpu() for x in args]

    def set_lr(self, lr, momentum=0.9):
        if lr == self.last_lr:
            return
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.last_lr = lr

    def train_batch(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        self.batch_size = len(inputs)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs.requires_grad = True
        outputs = self.model.forward(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.set_last(inputs, outputs, targets)
        loss = float(loss)
        self.steps.append(self.samples)
        self.losses.append(loss)
        self.batches += 1
        self.samples += len(inputs)
        return outputs.detach().softmax(1)

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return outputs.detach().softmax(1)

    def compute_error(self, inputs, targets):
        probs = self.predict(inputs).cpu()
        predicted = probs.argmax(1)
        return float((predicted != targets).sum()) / float(len(inputs))

    def plot_loss(self):
        import matplotlib.pyplot as plt
        import scipy.ndimage as ndi

        n = int(len(self.losses) ** 0.5) + 1
        xs = np.array(self.steps)[::n]
        ys = np.array(self.losses)[::n]
        graph = ndi.gaussian_filter(ys, 5.0, mode="nearest")
        plt.clf()
        plt.ion()
        plt.plot(xs, graph)
        plt.ginput(1, 0.0001)

    def __str__(self):
        return f"<Trainer {self.samples:10d} {np.mean(self.losses[-100:]):7.3e}>"


def make_loader(shards, batch_size=128, num_workers=6):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = (
        wds.WebDataset(shards)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(augment)
        .batched(batch_size)
    )
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None).repeat()
    return loader


def make_test_loader(shards, batch_size=512, num_workers=4):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augment = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    dataset = (
        wds.WebDataset(shards, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(augment)
        .batched(batch_size)
    )
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    return loader


def train(
    mname: str = "resnet18",
    device: str = "cuda:0",
    shards: str = "./data/imagenet-train-{000000..000146}.tar",
    rank: int = -1,
    size: int = -1,
    batch_size: int = 128,
    comms: str = "gloo",
    show: bool = False,
    testshards: str = "./data/imagenet-val-{000000..000006}.tar",
    test_batch_size: int = 1024,
    neval: int = 10,
    save_prefix: str = "",
    threads: int = -1,
    nworkers: int = 6,
    ntest_workers: int = 6,
):
    if size > 0:
        assert rank >= 0
        assert "MASTER_ADDR" in os.environ
        assert "MASTER_PORT" in os.environ
        dist.init_process_group(comms, rank=rank, world_size=size)
        distributed = True
        print(f"rank = {rank}")
    else:
        distributed = False
        print("single node")

    if save_prefix == "":
        save_prefix = mname

    if threads > 0:
        torch.set_num_threads(threads)

    loader = make_loader(shards, batch_size=batch_size, num_workers=nworkers)
    testloader = (
        make_test_loader(
            testshards, batch_size=test_batch_size, num_workers=ntest_workers
        )
        if testshards
        else None
    )

    models
    model = eval(f"models.{mname}")()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)

    trainer = Trainer(model)
    trainer.to(device)

    trigger_report = every(10, 0)
    trigger_eval = every(600, 0)

    print("starting")

    for inputs, targets in loader:

        trainer.train_batch(inputs, targets)

        if trigger_report():
            print(
                f"loss: {trainer.samples:10d} {np.mean(trainer.losses[-100:]):7.3e}",
                end="\r",
                flush=True,
            )
            if show:
                trainer.plot_loss()

        if trigger_eval():
            print()
            if rank <= 0:
                if testloader is not None:
                    print("evaluating")
                    last_err = np.mean(
                        [
                            trainer.compute_error(inputs, targets)
                            for inputs, targets in islice(testloader, 0, neval)
                        ]
                    )
                    print(f"test error: {trainer.samples:10d} {last_err}")
                fname = f"{save_prefix}-{trainer.samples//1000:06d}-{int(last_err*1e6):06d}.pth"
                print(f"saving {fname}")
                with open(fname, "wb") as stream:
                    torch.save(model.state_dict(), stream)


if __name__ == "__main__":
    typer.run(train)
