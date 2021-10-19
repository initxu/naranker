"""
Data: 2021/09/07
Target: 实现Xu_2021_ReNAS_CVPR中对nasbench101中网络的Feature Tensor encoding方法
Method: 将adjacency matrix, vertex的flops和params padding之后, 将每个cell的vertex flops list 和params list与
        adjacency matrix 相乘，构成两张代表这个cell的7*7 tensor, 全部结构共有9个cell共18张, 加上adjacency matrix
        共有19张7*7的tensor

Input: arch:dict
Output: torch.Tensor(C,H,W) default= 19*7*7 for nasbench101
"""

import torch
from  torch.nn import functional as F


type_dict = {'input':1, 'conv1x1-bn-relu':2 , 'conv3x3-bn-relu':3, 'maxpool3x3':4, 'output':5}

def adjacency_matrix_padding(matrix, arch_feature_dim, num_vertices):
   
    assert isinstance(matrix, torch.Tensor), 'adjacency matrix type should be torch.Tensor'
    
    pad_matrix = matrix

    for _ in range(arch_feature_dim - num_vertices):
        
        pd = (0,1,0,1)                                  #前两个0,1是沿着最后一个dim(列索引)的两边pad 0列和1列，后两个0，1沿着倒数第二个dim的两边pad 0行和1行
        pad_matrix= F.pad(pad_matrix, pd, 'constant', 0)
        
        index = [i for i in range(len(pad_matrix))]     # pad增加一列后矩阵的列索引    
        index[-2], index[-1] = index[-1], index[-2]     # 交换最后两行的索引 [0,1,2] → [0,2,1]
        
        pad_matrix = pad_matrix[index]                  # 根据索引交换最后两行 matrix = matrix[[0,2,1]]
        pad_matrix = pad_matrix.transpose(0,1)          # 转置矩阵，将不可操作的列转置为行以便交换
        pad_matrix = pad_matrix[index]                  # 交换行
        pad_matrix = pad_matrix.transpose(0,1)          # 交换完后转置还原
    
    return pad_matrix

def vector_padding(vector, arch_feature_dim, num_vertices):
    
    assert isinstance(vector, torch.Tensor), 'vector type should be torch.Tensor'

    pad_vector = vector
    
    for _ in range(arch_feature_dim - num_vertices):
        
        pd = (0,1)
        pad_vector = F.pad(pad_vector, pd, 'constant', 0)       # 在vector最后一维pad一个0

        index = [i for i in range(len(pad_vector))]
        index[-2], index[-1] = index[-1], index[-2]
        pad_vector = pad_vector[index]

    return pad_vector

def feature_tensor_encoding(arch:dict, arch_feature_dim=7, arch_feature_channels=19):
        
    matrix = arch['module_adjacency']
    ops = arch['module_operations']
    # params_ = arch['trainable_parameters']
    # flops_ = arch['flops']
    vertex_flops_dict_ = arch['vertex_flops']
    vertex_params_dict_ = arch['vertex_params']

    num_vertices = len(matrix)
    
    ops_vector = torch.tensor([type_dict[v] for v in ops])     # 算子的编码
    matrix = torch.tensor(matrix)      # HW, dim=0是行索引，dim=1是列索引
    
    if num_vertices < arch_feature_dim:
        matrix = adjacency_matrix_padding(matrix, arch_feature_dim, num_vertices)
        ops_vector = vector_padding(ops_vector, arch_feature_dim, num_vertices)
    
    ops_vector_matrix = torch.mul(matrix,ops_vector)
    
    arch_feature = ops_vector_matrix.unsqueeze(dim=0)      # arch_feature_tensor: HW → CHW, 第一维是通道，第二维是行索引，第三维是列索引
    
    for _, (fk,pk) in enumerate(zip(vertex_flops_dict_,vertex_params_dict_)):
        
        cell_flops = torch.tensor(vertex_flops_dict_[fk])
        cell_params = torch.tensor(vertex_params_dict_[pk])

        cell_flops = torch.true_divide(cell_flops, 1e7)
        cell_params = torch.true_divide(cell_params, 1e5)

        if num_vertices < arch_feature_dim:
            cell_flops = vector_padding(cell_flops, arch_feature_dim, num_vertices)
            cell_params = vector_padding(cell_params, arch_feature_dim, num_vertices)

        cell_flops_matrix = torch.mul(matrix,cell_flops).unsqueeze(dim=0)       # 2个操作: 首先与matrix点乘构成矩阵，其次增加通道维度HW → CHW
        cell_params_matrix = torch.mul(matrix, cell_params).unsqueeze(dim=0)
        
        arch_feature = torch.cat([arch_feature,cell_flops_matrix,cell_params_matrix],dim=0)

    assert len(arch_feature) == arch_feature_channels, 'Wrong channels of arch feature tensor'
    return arch_feature


if __name__ == "__main__":

    # for debug function [adjacency_matrix_padding]
    # a = [[2, 2, 2, 9], 
    #     [2, 2, 2, 9], 
    #     [2, 2, 2, 9], 
    #     [9, 9, 9, 9],]
    # a = torch.tensor(a)
    # a = adjacency_matrix_padding(a,7,len(a))
    # print(a)

    #  for debug function [vector_padding]
    # a = [2, 2, 2, 9]
    # a = torch.Tensor(a)
    # a = vector_padding(a, 7, len(a))
    # print(a)


    # for test,测试从json中提出list存储的arch
    import json
    data_path = '/home/ubuntu/workspace/nar/target.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    f.close()
    assert isinstance(dataset,list)
    arch = dataset[1]

    encoded_arch = feature_tensor_encoding(arch)
    print(encoded_arch)
    print(encoded_arch.shape)