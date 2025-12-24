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


def calculate_metrics(arrays_list, loader):
    metrics = []
    for arrays in arrays_list:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(DEVICE)

        loss, acc = test_fn(model, loader, DEVICE)
        metrics.append((loss, acc))

    return metrics


def select_best_models(metrics, thresh=0.9):
    losses = [loss for loss, acc in metrics]
    maxl = max(losses)
    minl = min(losses)
    dl = maxl - minl
    weights = len(metrics) * [0]

    if dl == 0:
        weights[i] = len(metrics) * [1]
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
    metrics = calculate_metrics(arrays_list, evalloader)
    weights, selection = select_best_models(metrics)

    # fuse selection models weighed according to their loss
    fusion_np_arrays = {}
    for i in selection:
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
            METRICRECORD_IDENTITIES_KEY: selection,
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
    _, evalloader = load_data(partition_id, num_partitions)
    metrics = calculate_metrics(arrays_list, evalloader)
    _, selection = select_best_models(metrics, thresh=1)
    # get its metrics to send in message
    loss, acc = metrics[selection[0]]

    # Construct and return reply Message
    content = RecordDict({
        METRICRECORD_KEY: MetricRecord({
            "eval_loss": loss,
            "eval_acc": acc,
            "identity": selection[0],
            METRICRECORD_WEIGHT_KEY: len(evalloader.dataset)
        })
    })
    return Message(content=content, reply_to=msg)
