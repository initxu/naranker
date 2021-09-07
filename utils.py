"""
Data: 2021/09/07
Target: 实现Xu_2021_ReNAS_CVPR中对nasbench101中网络的Feature Tensor encoding方法
Method: 将adjacency matrix, vertex的flops和params padding之后, 将每个cell的vertex flops list 和params list与
        adjacency matrix 相乘，构成两张代表这个cell的7*7 tensor, 全部结构共有9个cell共18张, 加上adjacency matrix
        共有19张7*7的tensor
"""

import json
import torch

from  torch.nn import functional as F

import numpy as np

def adjacency_matrix_padding(matrix, arch_feature_dim, num_vertices):
   
    assert isinstance(matrix, torch.Tensor), 'adjacency matrix type should be torch.Tensor'
    pad_matrix = matrix
    for _ in range(arch_feature_dim - num_vertices):
        
        pd = (0,1,0,1)                                  #前两个0,1是沿着最后一个dim(列索引)的两边pad 0列和1列，后两个0，1沿着倒数第二个dim的两边pad 0行和1行
        pad_matrix= F.pad(pad_matrix, pd, 'constant', 0)
        
        index = [i for i in range(len(pad_matrix))]     # 生成pad之后矩阵的列索引    
        index[-2], index[-1] = index[-1], index[-2]     # 交换最后两行的索引 [0,1,2] → [0,2,1]
        
        pad_matrix = pad_matrix[index]                  # 根据索引交换最后两行 matrix = matrix[[0,2,1]]
        pad_matrix = pad_matrix.transpose(0,1)          # 转置矩阵，将不可操作的列转置为行以便交换
        pad_matrix = pad_matrix[index]                  # 交换行
        pad_matrix = pad_matrix.transpose(0,1)          # 交换完后转置还原
    
    return pad_matrix

def feature_tensor_encoding(arch, arch_feature_dim=7, arch_feature_channels=19):
        
    matrix_ = arch['module_adjacency']
    ops_ = arch['module_operations']
    params_ = arch['trainable_parameters']
    flops_ = arch['flops']
    vertex_flops_dict_ = arch['vertex_flops']
    vertex_params_dict_ = arch['vertex_params']

    num_vertices = len(matrix_)
    
    matrix = torch.tensor(matrix_)      # HW, dim=0是行索引，dim=1是列索引
    if num_vertices < arch_feature_dim:
        matrix = adjacency_matrix_padding(matrix)

        

    
        
        
    # # arch_t = torch.tensor(matrix_).unsqueeze(dim=0)      # 加一维度的通道, HW → CHW, 第一维是通道，第二维是行索引，第三维是列索引
    
    
    # for i,keys in enumerate(zip(vertex_flops_dict,vertex_params_dict)):
    #     print(i,keys)

    #     import pdb;pdb.set_trace()







    return


if __name__ == "__main__":

    a = [
            [0, 1, 1, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 0, 1], 
            [0, 0, 0, 0, 0, 0]]
    a = torch.tensor(a)
    a = adjacency_matrix_padding(a,7,len(a))
    print(a)


    # for test,测试从json中提出list存储的arch
    data_path = '/home/ubuntu/workspace/nar/target.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    f.close()
    assert isinstance(dataset,list)
    arch = dataset[0]

    # TODO: 自定义一个arch，包含num_vertices < 7的各种情况

    encoded_tensors = feature_tensor_encoding(arch)