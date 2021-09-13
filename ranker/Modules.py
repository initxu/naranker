import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter 不是参数
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid)) # buffer:存储模型的状态state而不是需要更新梯度的参数，这里的位置编码是state而非模型的参数

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] # 每个位置position,对应一个d_hid=512维的向量

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])   # (200,512) 共200个位置，每个位置一个512长度的矢量
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()                                   # 将pos_table中存储的位置参数复制一份，与输入相加。detach方法将位置编码从当前计算图中剥离，因此将不会跟踪梯度

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: 256,8,36,64
        k: 256,8,36,64
        v: 256,8,36,64
        mask: 256,1,1,36
        """

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))    # (256,8,36,64)*(256,8,64,36)/T = (256,8,36,36) 矩阵乘法，前面的两维作为batch，后面的两维做矩阵乘法, 对应batch的二维矩阵乘，不同batch之间不会产生关系

        """
        mask(256,1,1,36)
        attn(256,8,36,36)
        这里用到了tensor的broadcast: 两个tensor必须满足，从最后一个维度开始往前算，维度要么相等，要么为1，要么不存在
        这里的mask中间两个维度为1，可以与attn做broadcast

        将mask的行索引复制到36，得到36×36的mask矩阵，batch中共256个36*36的矩阵，1/256即batch中每个样本的mask再复制到head=8个
        每个batch中样本的mask和各自的互注意力矩阵相乘
        注意力矩阵是36*36是个混淆矩阵，表示第一个元素和其余36个元素的关系，因此mask行列转置无所谓
        """
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)                    # 将mask=False的位置填0

        attn = self.dropout(F.softmax(attn, dim=-1))                    # 对混淆矩阵的最后一个维度做概率的归一:即每行代表序列中第i个词，与自己或其他序列中其他的词的相对关系的归一值
        output = torch.matmul(attn, v)                                  # [256, 8, 36, 36]*[256, 8, 36, 64] = [256, 8, 36, 64]

        return output, attn


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()                                                        # 获得序列大小
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()       # 对(1,len_s,len_s)矩阵保留上三角阵，diagonal=0表示保留对角线上元素，等于1表示对角线上元素也为0，1减去这个矩阵就是后面序列的mask
    """
    tensor([
        [[ True, False, False,  ..., False, False, False],
         [ True,  True, False,  ..., False, False, False],
         [ True,  True,  True,  ..., False, False, False],
         ...,
         [ True,  True,  True,  ...,  True, False, False],
         [ True,  True,  True,  ...,  True,  True, False],
         [ True,  True,  True,  ...,  True,  True,  True]]
         ], device='cuda:0')
         shape = 1,32,32
    """
    return subsequent_mask