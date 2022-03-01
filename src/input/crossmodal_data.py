# -*- coding: utf-8 -*-
from random import randint, seed
import numpy as np
from torch.utils.data import Dataset


data_src0 = np.tile([1] * 8 + [-1] * 8, 1).astype(np.float32)
data_src1 = np.tile([1] * 4 + [-1] * 4, 1).astype(np.float32)
data_src2 = np.tile([1] * 2 + [-1] * 2, 1).astype(np.float32)


def crossmodal_example_data(input_dim=16, length=32, delay=0):

    # spatial pattern params ####################
    # sp_pattern = {0: np.repeat([1, -1], 8).astype(np.float32),
    #               1: np.tile([1] * 4 + [-1] * 4, 2).astype(np.float32),
    #               2: np.tile([1] * 2 + [-1] * 2, 4).astype(np.float32)}
    n_rep0 = input_dim // 16
    n_rep1 = input_dim // 8
    n_rep2 = input_dim // 4
    sp_pattern0 = np.tile(data_src0, n_rep0 + 1)[0:input_dim]
    sp_pattern1 = np.tile(data_src1, n_rep1 + 1)[0:input_dim]
    sp_pattern2 = np.tile(data_src2, n_rep2 + 1)[0:input_dim]
    sp_pattern = {0: sp_pattern0, 1: sp_pattern1, 2: sp_pattern2}

    # temporal pattern params ###################
    tp_pattern = {0: 0.25, 1: 0.125, 2: 0.0625}

    # init data array
    data_array = np.zeros([input_dim, 1], dtype=np.float32)
    sp_label = []
    tp_label = []

    for sp in sp_pattern:
        for tp in tp_pattern:
            sp_base = sp_pattern[sp]
            tp_base = tp_pattern[tp]

            for t in range(length):
                tmp = sp_base * np.cos(2 * np.pi * tp_base * t)
                data_array = np.c_[data_array, tmp.reshape(input_dim, 1)]
                sp_label.append(sp)
                tp_label.append(tp)

    # temporal label delay (ignore label = -1)
    tp_label = [-1] * delay + tp_label
    tp_label = tp_label[:-delay]

    return data_array[:, 1:], np.array(sp_label, dtype=np.int32), np.array(tp_label, dtype=np.int32)


def crossmodal_random_sequence(input_dim=16, n_label=6, length=32, delay=0, input_seed=None):

    if input_seed is not None:
        seed(input_seed)

    # spatial pattern params ####################
    # sp_pattern = {0: np.repeat([1, -1], 8).astype(np.float32),
    #               1: np.tile([1] * 4 + [-1] * 4, 2).astype(np.float32),
    #               2: np.tile([1] * 2 + [-1] * 2, 4).astype(np.float32)}
    n_rep0 = input_dim // 16
    n_rep1 = input_dim // 8
    n_rep2 = input_dim // 4
    sp_pattern0 = np.tile(data_src0, n_rep0 + 1)[0:input_dim]
    sp_pattern1 = np.tile(data_src1, n_rep1 + 1)[0:input_dim]
    sp_pattern2 = np.tile(data_src2, n_rep2 + 1)[0:input_dim]
    sp_pattern = {0: sp_pattern0, 1: sp_pattern1, 2: sp_pattern2}

    # temporal pattern params ###################
    tp_pattern = {0: 0.25, 1: 0.125, 2: 0.0625}

    # init data array
    data_array = np.zeros([input_dim, 1], dtype=np.float32)
    sp_label = []
    tp_label = []

    t = 0.0
    for _ in range(n_label):
        sp = randint(0, 2)
        tp = randint(0, 2)
        sp_base = sp_pattern[sp]
        tp_base = tp_pattern[tp]

        for _ in range(length):
            tmp = sp_base * np.cos(2 * np.pi * tp_base * t)
            data_array = np.c_[data_array, tmp.reshape(input_dim, 1)]
            sp_label.append(sp)
            tp_label.append(tp)
            t += 1.0

    # temporal label delay (ignore label = -1)
    tp_label = [-1] * delay + tp_label
    tp_label = tp_label[:-delay]

    return data_array[:, 1:], np.array(sp_label, dtype=np.int32), np.array(tp_label, dtype=np.int32)


def crossmodal_random_sequence_random_change(input_dim=16, n_label=6, avg_length=32, delay=0, input_seed=None):

    if input_seed is not None:
        seed(input_seed)

    # spatial pattern params ####################
    # sp_pattern = {0: np.repeat([1, -1], 8).astype(np.float32),
    #               1: np.tile([1] * 4 + [-1] * 4, 2).astype(np.float32),
    #               2: np.tile([1] * 2 + [-1] * 2, 4).astype(np.float32)}
    n_rep0 = input_dim // 16
    n_rep1 = input_dim // 8
    n_rep2 = input_dim // 4
    sp_pattern0 = np.tile(data_src0, n_rep0 + 1)[0:input_dim]
    sp_pattern1 = np.tile(data_src1, n_rep1 + 1)[0:input_dim]
    sp_pattern2 = np.tile(data_src2, n_rep2 + 1)[0:input_dim]
    sp_pattern = {0: sp_pattern0, 1: sp_pattern1, 2: sp_pattern2}

    # temporal pattern params ###################
    tp_pattern = {0: 0.25, 1: 0.125, 2: 0.0625}

    total_length = int(n_label * avg_length)
    while True:
        change_time = [randint(1, total_length - 1) for _ in range(n_label - 1)]
        change_time = list(set(change_time))
        change_time.sort()
        if len(change_time) == (n_label - 1):
            break
    change_time = [0] + change_time + [total_length]
    print("change timing", change_time)

    # init data array
    data_array = np.zeros([input_dim, 1], dtype=np.float32)
    sp_label = []
    tp_label = []

    t = 0.0
    for start, end in zip(change_time[0:-1], change_time[1:]):
        sp = randint(0, 2)
        tp = randint(0, 2)
        sp_base = sp_pattern[sp]
        tp_base = tp_pattern[tp]

        for _ in range(end - start):
            tmp = sp_base * np.cos(2 * np.pi * tp_base * t)
            data_array = np.c_[data_array, tmp.reshape(input_dim, 1)]
            sp_label.append(sp)
            tp_label.append(tp)
            t += 1.0

    # temporal label delay (ignore label = -1)
    tp_label = [-1] * delay + tp_label
    tp_label = tp_label[:-delay]

    return data_array[:, 1:], np.array(sp_label, dtype=np.int32), np.array(tp_label, dtype=np.int32)

if __name__ == '__main__':

    print("Debug...")

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # debug for crossmodal_example_data #########################
    d, sp_l, tp_l = crossmodal_example_data()
    print("spatial:", sp_l)
    print("temporal:", tp_l)
    print()
    
    plt.figure(figsize=(12, 2))
    plt.imshow(d, cmap=plt.get_cmap('viridis'), interpolation='nearest')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("plot_example.pdf")

    # debug for crossmodal_random_sequence ######################
    d, sp_l, tp_l = crossmodal_random_sequence(n_label=6, length=32)
    print("spatial:", sp_l)
    print("temporal:", tp_l)
    print()
    
    plt.figure(figsize=(12, 2))
    plt.imshow(d, cmap=plt.get_cmap('viridis'), interpolation='nearest')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("plot_random_seq.pdf")

    # debug for crossmodal_random_sequence_random_change ########
    d, sp_l, tp_l = crossmodal_random_sequence_random_change(n_label=6, avg_length=32)
    print("spatial:", sp_l)
    print("temporal:", tp_l)
    print()
    
    plt.figure(figsize=(12, 2))
    plt.imshow(d, cmap=plt.get_cmap('viridis'), interpolation='nearest')
    plt.clim(-1, 1)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("plot_random_seq_rand_change.pdf")
    