from flwr.common import Array, ArrayRecord

import torch


def pack_list_arrays(list_arrays: list[ArrayRecord]) -> ArrayRecord:
    list_arrays_packed = ArrayRecord()
    i = 0
    for arrays in list_arrays:
        for key, value in arrays.items():
            indexed_key = f"{i} {key}"
            list_arrays_packed[indexed_key] = value
        i += 1

    return list_arrays_packed


def unpack_list_arrays(list_arrays_packed: ArrayRecord) -> list[ArrayRecord]:
    list_arrays = []
    for indexed_key, value in list_arrays_packed.items():
        i_str, key = indexed_key.split(" ")
        i = int(i_str)

        while len(list_arrays) <= i:
            list_arrays.append(ArrayRecord())

        list_arrays[i][key] = value

    return list_arrays
