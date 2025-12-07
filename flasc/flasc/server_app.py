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
    num_global_models: int = context.run_config["num-global-models"]

    # Create a list of C models
    state_dict_list = [Net().state_dict() for _ in range(num_global_models)]
    # index each model and concatenate as if they were a single model
    state_dict_idx = concat_state_dict_list(state_dict_list)
    # create array record of amalgamate
    arrays = ArrayRecord(array_dict=state_dict_idx)

    # Initialize FedAvg strategy
    strategy = FLASC(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "num-global-models": num_global_models}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final models to disk...")
    # big change here
    # state_dict = result.arrays.to_torch_state_dict()
    # torch.save(state_dict, "final_model.pt")
