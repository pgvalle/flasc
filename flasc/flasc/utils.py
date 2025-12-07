import torch
from flwr.common import Array
from collections import OrderedDict


def concat_state_dict_list(state_dict_list):
    idx_state_dict = OrderedDict()
    i = 0
    for state_dict in state_dict_list:
        for key, value in state_dict.items():
            idx_key = f"{i}.{key}"
            idx_state_dict[idx_key] = Array.from_torch_tensor(value)
        i += 1

    return idx_state_dict

def sep_idx_state_dict(state_dict_idx):
    state_dict_list = []
    for idx_key, array in state_dict_idx.items():
        subs = idx_key.split(".")
        idx = int(subs.pop(0))
        key = ".".join(subs)

        if len(state_dict_list) == idx:
            state_dict_list.append(OrderedDict())

        state_dict_list[idx][key] = torch.from_numpy(array.numpy())

    return state_dict_list
