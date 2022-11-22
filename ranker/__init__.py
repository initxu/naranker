"""
Data: 2021/09/13
Target: 结构对比和排名预测器
Method: 利用transformer, Encoder输入采样结构的编码，对邻接矩阵pad行、列进行mask，embedding后送入自注意力和positionfeedforward进行结构序列的注意力关系的提取，作为decoder的K，V输入
        Decoder首先对5个tier的编码做特征提取作为Q，然后与K，V做cross注意力，匹配Q和K，V的相似性(即采样结构和tier的关系)，输出概率和对应的表征

Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
"""

from .models import Transformer
from .models_q2l import TransformerQ2L
from .models_latency import  TransformerLatency
from .models_latency_q2l import  TransformerLatencyQ2L

__all__ = [Transformer, TransformerLatency,TransformerLatencyQ2L]
