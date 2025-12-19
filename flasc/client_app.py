"""FLASC: A Flower / PyTorch app."""
from flwr.clientapp import ClientApp
from flwr.common import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
    Array,
)

from flasc.task import (
    Net,
    load_data,
    test as test_fn,
    train as train_fn
)

from flasc.strategy import (
    FLASC,
    METRICRECORD_WEIGHT_KEY,
    METRICRECORD_IDENTITIES_KEY,
    ARRAYRECORD_KEY,
    CONFIGRECORD_KEY,
    METRICRECORD_KEY
)

import torch
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Flower ClientApp
app = ClientApp()


def select_best_models(arrays_list, loader, thresh=0.9):
    losses = []
    for arrays in arrays_list:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(DEVICE)

        loss, _ = test_fn(model, loader, DEVICE)
        losses.append(loss)

    maxl = max(losses)
    minl = min(losses)
    dl = maxl - minl
    weights = None

    if dl == 0:
        import random
        i = random.randint(0, len(arrays_list) - 1)
        weights = len(arrays_list) * [0]
        weights[i] = 1
    else:
        # weight[i] = 1 - normalized_loss[i]
        weights = [1 - (l - minl) / dl for l in losses]

    selection = []
    for i in range(len(weights)):
        if weights[i] >= thresh:
            selection.append(i)

    return weights, selection


@app.train()
def train(msg: Message, context: Context):
    """Evaluate each cluster model on local data,
    fuse the best ones and train the fusion on local data"""
    
    # Load the local data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, evalloader = load_data(partition_id, num_partitions)

    # load amalgamate of state_dicts and convert to list of state_dicts
    packed = msg.content[ARRAYRECORD_KEY]
    arrays_list = FLASC.unpack_arrays(packed)

    # select the best models
    weights, selected = select_best_models(arrays_list, evalloader, thresh=0.9)

    # fuse selected models weighed according to their loss
    fusion_np_arrays = {}
    for i in selected:
        arrays = arrays_list[i]
        weight = weights[i]
        for k, v in arrays.items():
            if k not in fusion_np_arrays:
                fusion_np_arrays[k] = v.numpy() * weight
            else:
                fusion_np_arrays[k] += v.numpy() * weight

    # load fusion as pytorch model
    fusion_arrays = ArrayRecord({
        k: Array(np.asarray(v)) for k, v in fusion_np_arrays.items()
    })
    fusion = Net()
    fusion.load_state_dict(fusion_arrays.to_torch_state_dict())

    # train fused model
    train_loss = train_fn(
        fusion,
        trainloader,
        context.run_config["local-epochs"],
        msg.content[CONFIGRECORD_KEY]["lr"],
        DEVICE
    )

    # Construct and return reply Message
    content = RecordDict({
        ARRAYRECORD_KEY: fusion_arrays,
        METRICRECORD_KEY: MetricRecord({
            "train_loss": train_loss,
            METRICRECORD_IDENTITIES_KEY: selected,
            METRICRECORD_WEIGHT_KEY: len(trainloader.dataset),
        })
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context): #! modificar para treinar cada modelo de cluster
    """Evaluate each cluster model on local data."""

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, evalloader = load_data(partition_id, num_partitions)

    # load amalgamate of state_dicts and convert to list of state_dicts
    packed = msg.content[ARRAYRECORD_KEY]
    arrays_list = FLASC.unpack_arrays(packed)

    # select the best model
    _, selected = select_best_models(arrays_list, evalloader, thresh=1)
    i = selected[0]

    # load it as a pytorch model
    arrays = arrays_list[i]
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    # evaluate it
    loss, acc = test_fn(model, evalloader, DEVICE)

    # Construct and return reply Message
    content = RecordDict({
        METRICRECORD_KEY: MetricRecord({
            "eval_loss": loss,
            "eval_acc": acc,
            "identity": i,
            METRICRECORD_WEIGHT_KEY: len(evalloader.dataset)
        })
    })
    return Message(content=content, reply_to=msg)
