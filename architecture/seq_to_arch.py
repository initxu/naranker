import numpy as np

DUMMY = 'dummy'
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

NODE_TYPE = [DUMMY, INPUT, CONV1X1, CONV3X3, MAXPOOL3X3, OUTPUT]


def seq_to_arch(seq):
    n_nodes = seq[0]

    opt = [INPUT]
    matrix = np.zeros(shape=(n_nodes, n_nodes), dtype=np.int32)

    output_node = seq[2 * (n_nodes - 2) + 1]  # 第output节点的下标 2*(n_nodes-2)+1
    inter_nodes = seq[1:2 * (n_nodes - 2) + 1]
    inter_nodes_edges = inter_nodes[::2]
    inter_nodes_type = inter_nodes[1::2]
    remain_edges = seq[2 * (n_nodes - 2) + 2:]

    # 中间节点的edge和type
    for end_node, (begin_node,
                   opt_type) in enumerate(zip(inter_nodes_edges,
                                              inter_nodes_type),
                                          start=1):
        opt.append(NODE_TYPE[opt_type])
        matrix[begin_node][end_node] = 1

    # output节点的edge和type
    opt.append(OUTPUT)
    matrix[output_node][n_nodes - 1] = 1

    # remain edges
    re_start_ndoes = remain_edges[::2]
    re_end_nodes = remain_edges[1::2]
    for (begin_node, end_node) in zip(re_start_ndoes, re_end_nodes):
        if begin_node == None or end_node == None:
            continue
        matrix[begin_node][end_node] = 1

    return matrix, opt


if __name__ == '__main__':
    # for debug
    # 输入是[66, 0, 2, 1, 4, 0, 4, 0, 2, 0, 2, 2, None, None, None, None, 2, 4], length = max_edges=9 for nasbench101
    seq_7 = [7, 0, 4, 0, 2, 2, 2, 3, 2, 0, 2, 5, 0, 1, 2, 4, 0, 2]
    seq_5 = [5, 0, 2, 0, 2, 0, 3, 3, 0, 3, 3, 4, 2, 3, None, None, 3, 4]
    seq__77 = [7, 0, 4, 1, 3, 0, 4, 2, 3, 2, 4, 1, 4, 6, 2, 5, None, None]
    print(seq_to_arch(seq_5))
