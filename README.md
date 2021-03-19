Code illustrating distributed deep learning using WebDataset for data loading.

The default configuration uses shards stored in Google Cloud.

Start by installing the necessary virtual environment:

    $ ./run venv

If you don't have it, install `curl`:

    $ sudo apt-get install curl

# `distimgnet.py`

This illustrates training on ImageNet shards. You need to generate the shards
from the original Imagenet data (the script is part of the WebDataset distribution).

The recommended way of training with many workers and on large datasets is to
dispense with epochs altogether and just use loading like this:

    dataset = wds.WebDataset(shards).shuffle(1000) ... .batched(batch_size, partial=False)
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None).repeat()

    for inputs, targets in loader:
        trainer.train_batch(inputs, targets)

`WebLoader` is just a thin wrapper around `DataLoader` that makes available some
convenience methods.

Note that we do the batching inside each worker; this is important for efficiency,
since transferring individual samples is slow. However, the worker batch size isn't
necessarily the batch size you finally want, and furthermore, it's a good idea to
shuffle between different workers. For this, you can write:

    dataset = wds.WebDataset(shards).shuffle(1000) ... .batched(worker_batch_size)
    loader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    loader = loader.unbatched().shuffle(1000).batched(batch_size)

    for inputs, targets in loader:
        trainer.train_batch(inputs, targets)

(more documentation to follow)

# `simdist.py`

The recommended way for multinode/large data training is without epochs, as
illustrated above.

However, if you want to use training-by-epochs together with multinode
distributed data parallel training, you need to ensure that each DataLoader
on every node returns exactly the same number of batches.

You can only do this exactly if you divide your dataset into a number
of shards that is divisible by the product of the number of nodes and the
number of workers.

If your data isn't in this format, you may have to either drop some batches
or repeat some batches. The `simdist.py` example shows how you can do this.
Note that the dropping/repeating has fairly little influence, since the way
these pipelines are structured, different samples will be dropped/repeated each
epoch.

Use `run simdist.py --help` to see the available options.
Look at the source code to see how the flags are used to configure the pipeline.
Use the `--showsplits` and `--showopen` options to see information about when/how
shards are split/shuffled and when shards are opened.

This fails:

    $ ./run simdist.py

This works:

    $ ./run simdist.py --nbatches 26
    $ ./run simdist.py --nbatches 32 --dsrepeat 2
