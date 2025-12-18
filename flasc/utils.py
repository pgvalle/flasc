from flwr.common import Array, ArrayRecord

import torch


def pack_state_dicts(state_dicts: list[ArrayRecord]) -> ArrayRecord:
    state_dicts_packed = ArrayRecord()
    i = 0
    for state_dict in state_dicts:
        for key, value in state_dict.items():
            indexed_key = f"{i} {key}"
            array = Array.from_torch_tensor(value)
            state_dicts_packed[indexed_key] = array
        i += 1

    return state_dicts_packed


def unpack_state_dicts(state_dicts_packed: ArrayRecord) -> list[ArrayRecord]:
    state_dicts = []
    for indexed_key, array in state_dicts_packed.items():
        i_str, key = indexed_key.split(" ")
        i = int(i_str)

        if len(state_dicts) == i:
            state_dicts.append({})

        state_dict = torch.from_numpy(array.numpy())
        state_dicts[i][key] = state_dict

    return state_dicts
