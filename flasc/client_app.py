"""FLASC: A Flower / PyTorch app."""
import sys
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict
)
from flwr.clientapp import ClientApp

from flasc.task import Net, load_data
from flasc.task import test as test_fn
from flasc.task import train as train_fn
from flasc.utils import unpack_state_dicts


# Flower ClientApp
app = ClientApp()


def select_best_models(losses, thresh=0.9):
    """ The lower the loss, the better the model is. """

    maxl = max(losses)
    minl = min(losses)
    dl = maxl - minl
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the local data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, evalloader = load_data(partition_id, num_partitions)

    # load amalgamate of state_dicts and convert to list of state_dicts
    state_dicts_packed = msg.content["arrays"]
    state_dicts = unpack_state_dicts(state_dicts_packed)

    # load each model from state_dict_list and calculate their losses
    losses = []
    for state_dict in state_dicts:
        model = Net()
        model.load_state_dict(state_dict)
        model.to(device)

        loss, _ = test_fn(model, evalloader, device)
        losses.append(loss)

    # select the best models
    weights, selected = select_best_models(losses, thresh=0.9)

    # fuse selected models weighed according to their loss
    fusion_state_dict = {}
    for i in selected:
        state_dict = state_dicts[i]
        weight = weights[i]
        for key, value in state_dict.items():
            if key not in fusion_state_dict:
                fusion_state_dict[key] = value * weight
            else:
                fusion_state_dict[key] += value * weight

    # load fusion as pytorch model
    fusion = Net()
    fusion.load_state_dict(fusion_state_dict)

    # train fused model
    train_loss = train_fn(
        fusion,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device
    )

    # Construct and return reply Message
    content = RecordDict({
        "arrays": ArrayRecord(fusion_state_dict),
        "config": ConfigRecord({"identities": selected}),
        "metrics": MetricRecord({
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset)
        })
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context): #! modificar para treinar cada modelo de cluster
    """Evaluate each cluster model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, evalloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        evalloader,
        device,
    )

    # Construct and return reply Message
    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(evalloader.dataset),
        })
    })
    return Message(content=content, reply_to=msg)
