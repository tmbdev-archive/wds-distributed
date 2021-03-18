import sys
import os
import warnings
import torch
import webdataset as wds
import typer
import braceexpand
from collections import Counter
from itertools import islice
from torchvision import transforms


# We're not using actual torch.distributed, since we just want to simulate
# how data is split between different nodes. Other than that, though, this
# code works the same way as true distributed code.

dist_rank = -1
dist_size = -1

show_splits = False


def split_by_node(urls):

    """Split urls for each node.

    This uses the rank and world size. Note that it is invoked in each worker,
    so the results need to be consistent between multiple invocations."""

    global dist_rank, dist_size
    if dist_rank >= 0 and dist_size > 0:
        result = urls[dist_rank::dist_size]
        if show_splits:
            print(
                f"split_by_node {dist_rank}/{dist_size} len={len(result)}",
                file=sys.stderr,
            )
        return result
    else:
        print(f"single node len={len(result)}")
        return urls


def split_by_worker(urls):

    """Split urls for each worker."""

    urls = [url for url in urls]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        if wid == 0 and len(urls) < num_workers:
            warnings.warn(f"num_workers {num_workers} > num_shards {len(urls)}")
        result = urls[wid::num_workers]
        if show_splits:
            print(
                f"split_by_worker {wid}/{num_workers} len={len(result)}",
                file=sys.stderr,
            )
        return result
    else:
        return urls


def make_loader(shards, batch_size=128, num_workers=6, partial=False, repeat=1):

    """Create a loader for Imagenet-like data.

    The `partial` argument is passed on to the `batched()` method.
    Note that if `partial` is True, each worker may return a partial batch."""

    augment = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]
    )
    dataset = (
        wds.WebDataset(shards, nodesplitter=split_by_node, splitter=split_by_worker)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(augment)
        .batched(batch_size, partial=partial)
    )
    if repeat > 1:
        dataset = dataset.repeat(repeat)
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    return loader


def train(
    shards: str = "pipe:gsutil cat gs://lpr-simsplit/split-{000000..00009}.tar",
    size: int = 3,
    batch_size: int = 10,
    nworkers: int = 3,
    nepochs: int = 1,
    nbatches: int = 999999,
    partial: bool = False,
    showopen: bool = False,
    showsplits: bool = False,
    repeat: int = 1,
    dsrepeat: int = 1,
):
    """Simulate distributed training.

    This will perform dataset loading for each worker in a distributed training
    job of size `size` and report the number of batches and samples returned by
    each worker.

    For distributed SGD (DistributedDataParallel) to work, each worker needs to return
    exactly the same number of batches. To get exact epochs, you need to ensure that
    all the shards have exactly the same number of samples and that the number of shards
    is divisible by (#workers * #nodes).

    If your data isn't in that form (and it usually isn't), you have to do something different.
    """

    print("parameters:")
    print(f"\tworldsize {size}")
    print(f"\tnworkers {nworkers}")
    print(f"\tnshards {len(list(braceexpand.braceexpand(shards)))}")
    print(f"\tpartial {partial}")
    print(f"\tloader-repeat {repeat}")
    print(f"\tdataset-repeat {dsrepeat}")
    print()

    global dist_size, dist_rank, show_splits

    show_splits = showsplits

    if showopen:
        os.environ["GOPEN_VERBOSE"] = "1"

    dist_size = size

    loader = make_loader(
        shards,
        batch_size=batch_size,
        num_workers=nworkers,
        partial=partial,
        repeat=dsrepeat,
    )
    if repeat > 1:
        loader = loader.repeat(nepochs=repeat)

    batches = []
    for rank in range(size):
        dist_rank = rank
        batches.append([])
        for inputs, targets in islice(loader, 0, nbatches):
            batches[-1].append(len(inputs))

        # print(f"=== rank {dist_rank} batches {len(batches[-1])} total {np.sum(batches[-1])}")
        print(f"rank {rank}:", Counter(batches[-1]).most_common())

    counted = [tuple(Counter(x).most_common()) for x in batches]

    if not all([c == counted[0] for c in counted]):
        print("\nFAILED: inconsistent batches in different workers")
    else:
        print("\nSUCCESS: same batches in all workers")


if __name__ == "__main__":
    typer.run(train)
