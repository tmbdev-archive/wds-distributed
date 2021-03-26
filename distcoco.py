import os
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import webdataset as wds
import numpy as np
import time
import typer
from itertools import islice
import scipy.ndimage as ndi
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchmore import flex
from torchmore import combos


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


def sigloss(outputs, targets):
    return F.mse_loss(torch.sigmoid(outputs), targets)


class Trainer:
    def __init__(self, model, schedule=None):
        self.model = model
        self.criterion = sigloss
        self.last_lr = None
        self.device = "cpu"
        self.clip_grad = 1.0
        self.batches = 0
        self.samples = 0
        self.set_lr(0.1)
        self.steps = []
        self.losses = []
        self.schedule = schedule

    def to(self, device):
        self.device = device
        self.model.to(device)

    def set_last(self, *args):
        self.last = [x.detach().cpu() for x in args]

    def set_lr(self, lr, momentum=0.9):
        if lr == self.last_lr:
            return
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.last_lr = lr

    def train_batch(self, inputs, targets):
        if self.schedule is not None:
            self.set_lr(self.schedule(self.samples))
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

    def probs(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return torch.sigmoid(outputs)

    def compute_error(self, inputs, targets):
        probs = self.probs(inputs).cpu()
        assert probs.shape == targets.shape
        denom = targets.sum(1).clip(1., 1e30)
        errs = (probs - targets).abs().sum(1) / denom
        return errs.mean()

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


def keep_only(*args):
    def f(s):
        return {arg: s[arg] for arg in args}

    return f


def make_sample(sample):
    image = sample["jpg"]
    info = sample["json"]
    h, w, d = image.shape
    assert h > 10 and h <= 640 and w > 10 and w <= 640 and d == 3 and image.dtype == np.uint8
    image = image.astype(float) / 256.0
    scale = max(w, h) / 256.0
    image = ndi.affine_transform(
        image, np.diag([scale, scale, 1]), output_shape=(256, 256, 3), order=1, mode="constant"
    )
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float)
    target = torch.zeros(256, dtype=torch.float)
    if "annotations" in info:
        for a in info["annotations"]:
            target[a["category_id"]] = 1.0
    return image, target


def make_loader(shards, batch_size=128, num_workers=6, mode="train"):
    dataset = (
        wds.WebDataset(shards)
        .shuffle(1000)
        .map(keep_only("jpg", "json"))
        .decode("rgb8")
        .map(make_sample)
        .batched(batch_size)
    )
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None).repeat()
    return loader


def make_model(noutput=256, sizes=[32, 64, 128, 256, 512], shape=(3, 3, 256, 256), device="cpu"):
    blocks = []
    for size in sizes:
        blocks += combos.conv2d_block(size, repeat=2, mp=2)
    model = nn.Sequential(
        *blocks,
        nn.Flatten(),
        flex.Linear(1024),
        flex.BatchNorm(),
        nn.ReLU(),
        flex.Linear(1024),
        flex.BatchNorm(),
        nn.ReLU(),
        flex.Linear(256),
    )
    flex.shape_inference(model, shape)
    return model


def train(
    mname: str = "coco",
    device: str = "cuda:0",
    shards: str = "http://storage.googleapis.com/lpr-coco2017/coco-train-{000000..000118}.tar",
    testshards: str = "http://storage.googleapis.com/lpr-coco2017/coco-val-{000000..000003}.tar",
    rank: int = -1,
    size: int = -1,
    batch_size: int = 64,
    comms: str = "gloo",
    show: bool = False,
    test_batch_size: int = -1,
    neval: int = 10,
    save_prefix: str = "",
    threads: int = -1,
    nworkers: int = 12,
    ntest_workers: int = 12,
    schedule: str = "0.1 / (1+n//100000)**.5",
    display: float = 1e30,
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

    if test_batch_size < 0:
        test_batch_size = batch_size

    loader = make_loader(shards, batch_size=batch_size, num_workers=nworkers)
    if testshards != "":
        testloader = (
            make_loader(testshards, batch_size=test_batch_size, num_workers=ntest_workers, mode="test")
            if testshards
            else None
        )
    else:
        testloader = None

    model = make_model()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)
    print(model)

    trainer = Trainer(model, schedule=eval(f"lambda n: {schedule}"))
    trainer.to(device)

    trigger_report = every(10, 0)
    trigger_eval = every(600, 0)
    display_trigger = every(display)

    print("starting")

    for inputs, targets in loader:

        assert inputs.ndimension() == 4, inputs.shape

        if display_trigger():
            plt.clf()
            plt.imshow(inputs[0].numpy().transpose(1, 2, 0))
            plt.ginput(1, 0.001)

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
            if rank <= 0 and testloader is not None:
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
