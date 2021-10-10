"""
Date: 2020/10/09
Target: encode NasBench201 arch into 31*4*4 feature tensor for each set in \
        {'cifar10-valid', 'cifar100', 'ImageNet16-120'}
"""

import torch
import copy

from .nasbench201 import str2lists


def feature_tensor_encoding_201(arch: dict,
                                arch_feature_dim=4,
                                arch_feature_channels=31):

    matrix = arch['cell_adjacency']
    assert len(matrix) == arch_feature_dim, 'Wrong length of adjacency matrix'
    matrix = torch.tensor(matrix)

    arch_str = arch['arch_str']
    arch_opt_list = str2lists(arch_str)
    coordi_list = []
    for col_id, node_ops in enumerate(arch_opt_list, start=1):
        for op in node_ops:
            coordi_list.append([op[1], col_id])
    # [start_node, end_node] [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]

    all_type_tensors_list = {}

    for net_type in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
        opt_flops = arch['{}_opt_flops'.format(net_type)]
        opt_params = arch['{}_opt_params'.format(net_type)]
        feature_tensor_list = []
        feature_tensor_list.append(copy.deepcopy(matrix).unsqueeze(dim=0))
        for cell_id, (flops, params) in enumerate(zip(opt_flops.values(), opt_params.values())):
            f_patch = torch.zeros(arch_feature_dim, arch_feature_dim)
            p_patch = torch.zeros(arch_feature_dim, arch_feature_dim)
            for edge_id, (coord, edge_flops, edge_params) in enumerate(zip(coordi_list, flops, params)):
                f_patch[coord[0]][coord[1]] = edge_flops
                p_patch[coord[0]][coord[1]] = edge_params
            
            feature_tensor_list.append(f_patch.unsqueeze(dim=0))
            feature_tensor_list.append(p_patch.unsqueeze(dim=0))
        
        arch_feature_tensor = torch.cat(feature_tensor_list, dim=0)
        assert arch_feature_tensor.size(0) == arch_feature_channels, 'Wrong arch feature_channels'
        
        all_type_tensors_list[net_type] = arch_feature_tensor

    return all_type_tensors_list


if __name__ == '__main__':
    a = {
        "cell_adjacency": [[0, 4, 2, 2], [0, 0, 1, 1], [0, 0, 0, 1],
                           [0, 0, 0, 0]],
        "cifar10_val_acc":
        81.98266665690103,
        "cifar10_test_acc":
        85.86333333333334,
        "cifar100_val_acc":
        52.70000000406901,
        "cifar100_test_acc":
        52.913333294677734,
        "imagenat16_val_acc":
        28.211111075507272,
        "imagenat16_test_acc":
        26.633333287556965,
        "cifar10_latency":
        0.0139359758611311,
        "cifar100_latency":
        0.013182918230692545,
        "image16_latency":
        0.012976543108622235,
        "cifar10-valid_total_flops":
        16315008,
        "cifar10-valid_total_params":
        129306,
        "cifar10-valid_opt_flops": {
            "cells0": [0, 294912, 0, 294912, 0, 0],
            "cells1": [0, 294912, 0, 294912, 0, 0],
            "cells2": [0, 294912, 0, 294912, 0, 0],
            "cells3": [0, 294912, 0, 294912, 0, 0],
            "cells4": [0, 294912, 0, 294912, 0, 0],
            "cells6": [0, 278528, 0, 278528, 0, 0],
            "cells7": [0, 278528, 0, 278528, 0, 0],
            "cells8": [0, 278528, 0, 278528, 0, 0],
            "cells9": [0, 278528, 0, 278528, 0, 0],
            "cells10": [0, 278528, 0, 278528, 0, 0],
            "cells12": [0, 270336, 0, 270336, 0, 0],
            "cells13": [0, 270336, 0, 270336, 0, 0],
            "cells14": [0, 270336, 0, 270336, 0, 0],
            "cells15": [0, 270336, 0, 270336, 0, 0],
            "cells16": [0, 270336, 0, 270336, 0, 0]
        },
        "cifar10-valid_opt_params": {
            "cells0": [0, 288, 0, 288, 0, 0],
            "cells1": [0, 288, 0, 288, 0, 0],
            "cells2": [0, 288, 0, 288, 0, 0],
            "cells3": [0, 288, 0, 288, 0, 0],
            "cells4": [0, 288, 0, 288, 0, 0],
            "cells6": [0, 1088, 0, 1088, 0, 0],
            "cells7": [0, 1088, 0, 1088, 0, 0],
            "cells8": [0, 1088, 0, 1088, 0, 0],
            "cells9": [0, 1088, 0, 1088, 0, 0],
            "cells10": [0, 1088, 0, 1088, 0, 0],
            "cells12": [0, 4224, 0, 4224, 0, 0],
            "cells13": [0, 4224, 0, 4224, 0, 0],
            "cells14": [0, 4224, 0, 4224, 0, 0],
            "cells15": [0, 4224, 0, 4224, 0, 0],
            "cells16": [0, 4224, 0, 4224, 0, 0]
        },
        "cifar100_total_flops":
        16320768,
        "cifar100_total_params":
        135156,
        "cifar100_opt_flops": {
            "cells0": [0, 294912, 0, 294912, 0, 0],
            "cells1": [0, 294912, 0, 294912, 0, 0],
            "cells2": [0, 294912, 0, 294912, 0, 0],
            "cells3": [0, 294912, 0, 294912, 0, 0],
            "cells4": [0, 294912, 0, 294912, 0, 0],
            "cells6": [0, 278528, 0, 278528, 0, 0],
            "cells7": [0, 278528, 0, 278528, 0, 0],
            "cells8": [0, 278528, 0, 278528, 0, 0],
            "cells9": [0, 278528, 0, 278528, 0, 0],
            "cells10": [0, 278528, 0, 278528, 0, 0],
            "cells12": [0, 270336, 0, 270336, 0, 0],
            "cells13": [0, 270336, 0, 270336, 0, 0],
            "cells14": [0, 270336, 0, 270336, 0, 0],
            "cells15": [0, 270336, 0, 270336, 0, 0],
            "cells16": [0, 270336, 0, 270336, 0, 0]
        },
        "cifar100_opt_params": {
            "cells0": [0, 288, 0, 288, 0, 0],
            "cells1": [0, 288, 0, 288, 0, 0],
            "cells2": [0, 288, 0, 288, 0, 0],
            "cells3": [0, 288, 0, 288, 0, 0],
            "cells4": [0, 288, 0, 288, 0, 0],
            "cells6": [0, 1088, 0, 1088, 0, 0],
            "cells7": [0, 1088, 0, 1088, 0, 0],
            "cells8": [0, 1088, 0, 1088, 0, 0],
            "cells9": [0, 1088, 0, 1088, 0, 0],
            "cells10": [0, 1088, 0, 1088, 0, 0],
            "cells12": [0, 4224, 0, 4224, 0, 0],
            "cells13": [0, 4224, 0, 4224, 0, 0],
            "cells14": [0, 4224, 0, 4224, 0, 0],
            "cells15": [0, 4224, 0, 4224, 0, 0],
            "cells16": [0, 4224, 0, 4224, 0, 0]
        },
        "ImageNet16-120_total_flops":
        4086272,
        "ImageNet16-120_total_params":
        136456,
        "ImageNet16-120_opt_flops": {
            "cells0": [0, 73728, 0, 73728, 0, 0],
            "cells1": [0, 73728, 0, 73728, 0, 0],
            "cells2": [0, 73728, 0, 73728, 0, 0],
            "cells3": [0, 73728, 0, 73728, 0, 0],
            "cells4": [0, 73728, 0, 73728, 0, 0],
            "cells6": [0, 69632, 0, 69632, 0, 0],
            "cells7": [0, 69632, 0, 69632, 0, 0],
            "cells8": [0, 69632, 0, 69632, 0, 0],
            "cells9": [0, 69632, 0, 69632, 0, 0],
            "cells10": [0, 69632, 0, 69632, 0, 0],
            "cells12": [0, 67584, 0, 67584, 0, 0],
            "cells13": [0, 67584, 0, 67584, 0, 0],
            "cells14": [0, 67584, 0, 67584, 0, 0],
            "cells15": [0, 67584, 0, 67584, 0, 0],
            "cells16": [0, 67584, 0, 67584, 0, 0]
        },
        "ImageNet16-120_opt_params": {
            "cells0": [0, 288, 0, 288, 0, 0],
            "cells1": [0, 288, 0, 288, 0, 0],
            "cells2": [0, 288, 0, 288, 0, 0],
            "cells3": [0, 288, 0, 288, 0, 0],
            "cells4": [0, 288, 0, 288, 0, 0],
            "cells6": [0, 1088, 0, 1088, 0, 0],
            "cells7": [0, 1088, 0, 1088, 0, 0],
            "cells8": [0, 1088, 0, 1088, 0, 0],
            "cells9": [0, 1088, 0, 1088, 0, 0],
            "cells10": [0, 1088, 0, 1088, 0, 0],
            "cells12": [0, 4224, 0, 4224, 0, 0],
            "cells13": [0, 4224, 0, 4224, 0, 0],
            "cells14": [0, 4224, 0, 4224, 0, 0],
            "cells15": [0, 4224, 0, 4224, 0, 0],
            "cells16": [0, 4224, 0, 4224, 0, 0]
        },
        "arch_str":
        "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|"
    }
    b = {
        "cell_adjacency": [[0, 3, 3, 1], [0, 0, 4, 3], [0, 0, 0, 1],
                           [0, 0, 0, 0]],
        "cifar10_val_acc":
        90.76933332438152,
        "cifar10_test_acc":
        93.66000000000001,
        "cifar100_val_acc":
        69.82666660970052,
        "cifar100_test_acc":
        70.29333330078126,
        "imagenat16_val_acc":
        44.48888875664605,
        "imagenat16_test_acc":
        44.033333248562286,
        "cifar10_latency":
        0.016852585892928276,
        "cifar100_latency":
        0.015363384176183631,
        "image16_latency":
        0.013773989677429198,
        "cifar10-valid_total_flops":
        114905728,
        "cifar10-valid_total_params":
        802426,
        "cifar10-valid_opt_flops": {
            "cells0": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells1": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells2": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells3": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells4": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells6": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells7": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells8": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells9": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells10": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells12": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells13": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells14": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells15": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells16": [2367488, 2367488, 0, 0, 2367488, 0]
        },
        "cifar10-valid_opt_params": {
            "cells0": [2336, 2336, 0, 0, 2336, 0],
            "cells1": [2336, 2336, 0, 0, 2336, 0],
            "cells2": [2336, 2336, 0, 0, 2336, 0],
            "cells3": [2336, 2336, 0, 0, 2336, 0],
            "cells4": [2336, 2336, 0, 0, 2336, 0],
            "cells6": [9280, 9280, 0, 0, 9280, 0],
            "cells7": [9280, 9280, 0, 0, 9280, 0],
            "cells8": [9280, 9280, 0, 0, 9280, 0],
            "cells9": [9280, 9280, 0, 0, 9280, 0],
            "cells10": [9280, 9280, 0, 0, 9280, 0],
            "cells12": [36992, 36992, 0, 0, 36992, 0],
            "cells13": [36992, 36992, 0, 0, 36992, 0],
            "cells14": [36992, 36992, 0, 0, 36992, 0],
            "cells15": [36992, 36992, 0, 0, 36992, 0],
            "cells16": [36992, 36992, 0, 0, 36992, 0]
        },
        "cifar100_total_flops":
        114911488,
        "cifar100_total_params":
        808276,
        "cifar100_opt_flops": {
            "cells0": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells1": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells2": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells3": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells4": [2392064, 2392064, 0, 0, 2392064, 0],
            "cells6": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells7": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells8": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells9": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells10": [2375680, 2375680, 0, 0, 2375680, 0],
            "cells12": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells13": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells14": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells15": [2367488, 2367488, 0, 0, 2367488, 0],
            "cells16": [2367488, 2367488, 0, 0, 2367488, 0]
        },
        "cifar100_opt_params": {
            "cells0": [2336, 2336, 0, 0, 2336, 0],
            "cells1": [2336, 2336, 0, 0, 2336, 0],
            "cells2": [2336, 2336, 0, 0, 2336, 0],
            "cells3": [2336, 2336, 0, 0, 2336, 0],
            "cells4": [2336, 2336, 0, 0, 2336, 0],
            "cells6": [9280, 9280, 0, 0, 9280, 0],
            "cells7": [9280, 9280, 0, 0, 9280, 0],
            "cells8": [9280, 9280, 0, 0, 9280, 0],
            "cells9": [9280, 9280, 0, 0, 9280, 0],
            "cells10": [9280, 9280, 0, 0, 9280, 0],
            "cells12": [36992, 36992, 0, 0, 36992, 0],
            "cells13": [36992, 36992, 0, 0, 36992, 0],
            "cells14": [36992, 36992, 0, 0, 36992, 0],
            "cells15": [36992, 36992, 0, 0, 36992, 0],
            "cells16": [36992, 36992, 0, 0, 36992, 0]
        },
        "ImageNet16-120_total_flops":
        28733952,
        "ImageNet16-120_total_params":
        809576,
        "ImageNet16-120_opt_flops": {
            "cells0": [598016, 598016, 0, 0, 598016, 0],
            "cells1": [598016, 598016, 0, 0, 598016, 0],
            "cells2": [598016, 598016, 0, 0, 598016, 0],
            "cells3": [598016, 598016, 0, 0, 598016, 0],
            "cells4": [598016, 598016, 0, 0, 598016, 0],
            "cells6": [593920, 593920, 0, 0, 593920, 0],
            "cells7": [593920, 593920, 0, 0, 593920, 0],
            "cells8": [593920, 593920, 0, 0, 593920, 0],
            "cells9": [593920, 593920, 0, 0, 593920, 0],
            "cells10": [593920, 593920, 0, 0, 593920, 0],
            "cells12": [591872, 591872, 0, 0, 591872, 0],
            "cells13": [591872, 591872, 0, 0, 591872, 0],
            "cells14": [591872, 591872, 0, 0, 591872, 0],
            "cells15": [591872, 591872, 0, 0, 591872, 0],
            "cells16": [591872, 591872, 0, 0, 591872, 0]
        },
        "ImageNet16-120_opt_params": {
            "cells0": [2336, 2336, 0, 0, 2336, 0],
            "cells1": [2336, 2336, 0, 0, 2336, 0],
            "cells2": [2336, 2336, 0, 0, 2336, 0],
            "cells3": [2336, 2336, 0, 0, 2336, 0],
            "cells4": [2336, 2336, 0, 0, 2336, 0],
            "cells6": [9280, 9280, 0, 0, 9280, 0],
            "cells7": [9280, 9280, 0, 0, 9280, 0],
            "cells8": [9280, 9280, 0, 0, 9280, 0],
            "cells9": [9280, 9280, 0, 0, 9280, 0],
            "cells10": [9280, 9280, 0, 0, 9280, 0],
            "cells12": [36992, 36992, 0, 0, 36992, 0],
            "cells13": [36992, 36992, 0, 0, 36992, 0],
            "cells14": [36992, 36992, 0, 0, 36992, 0],
            "cells15": [36992, 36992, 0, 0, 36992, 0],
            "cells16": [36992, 36992, 0, 0, 36992, 0]
        },
        "arch_str":
        "|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|"
    }
    l = feature_tensor_encoding_201(a)
    print(l)
