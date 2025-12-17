"""FLASC: A Flower / PyTorch app."""
import sys
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flasc.task import Net, load_data
from flasc.task import test as test_fn
from flasc.task import train as train_fn
from flasc.utils import *

# Flower ClientApp
app = ClientApp()

NET = Net() # reference model


def select_good_losses(losses, thresh):
    minl, maxl = min(losses), max(losses)
    dl = maxl - minl

    weights = [(l - minl) / dl for l in losses]
    pairs = [(weights[i], i) for i in range(len(weights))]
    pairs.sort(key=lambda pair: pair[0])

    return pairs[:3] # fixed n for now


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the local data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, evalloader = load_data(partition_id, num_partitions)

    # load amalgamate of state_dicts and convert to list of state_dicts
    state_dict_idx = msg.content["arrays"]
    state_dict_list = sep_idx_state_dict(state_dict_idx)

    # load each model from state_dict_list and calculate their losses
    models = []
    losses = []
    for state_dict in state_dict_list:
        model = Net()
        model.load_state_dict(state_dict)
        model.to(device)

        loss, acc = test_fn(model, evalloader, device)

        models.append(model)
        losses.append(loss)

    selection = select_good_losses(losses, thresh=0.1)
    # TODO: fedavg on all selected models
    for 
    for weight, index in selection:
        model = models[index]
    
    # TODO: evaluate each model with local data

    # TODO: fuse best fitting models
    # fused_model = ... 

    # train fused model
    train_loss = train_fn(
        fused_model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # TODO: add cluster identities in record
    fused_model_record = ArrayRecord(model.state_dict())
    cluster_record = ArrayRecord({"clusters": })
    metric_record = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    })

    content = RecordDict({
        "arrays": fused_model_record,
        "clusters": cluster_record
        "metrics": metric_record,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
