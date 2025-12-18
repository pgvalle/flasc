"""FLASC: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from flasc.task import Net
from flasc.utils import *
from flasc.strategy import FLASC

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    num_models: int = context.run_config["num-global-models"]

    # Create a list of C models
    state_dicts = [Net().state_dict() for _ in range(num_models)]
    # pack models as if they were a single model (this is easier than keeping them separate)
    state_dicts_packed = pack_state_dicts(state_dicts)
    # create array record of this amalgamate
    arrays = ArrayRecord(state_dicts_packed)

    # Initialize FedAvg strategy
    strategy = FLASC(num_models, fraction_train)
    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final models to disk...")
    # big change here
    # state_dict = result.arrays.to_torch_state_dict()
    # torch.save(state_dict, "final_model.pt")
