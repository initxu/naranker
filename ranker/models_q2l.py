import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers_q2l import EncoderLayer, DecoderLayer,Q2lDecoderLayer
from .modules import PositionalEncoding
import math

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_patch_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(d_patch_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        src_seq = self.dropout(src_seq)
        enc_output = self.position_enc(src_seq)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_patch_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(d_patch_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        trg_seq = self.dropout(trg_seq)
        dec_output = self.position_enc(trg_seq)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Q2lDecoder(nn.Module):
    ''' A decoder model with self attention mechanism for Q2l. '''

    def __init__(
            self, d_patch_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(d_patch_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            Q2lDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask,query_pos, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward v3
        trg_seq = self.dropout(trg_seq)
        dec_output = self.position_enc(trg_seq)
        dec_output = self.layer_norm(dec_output)
        dec_output=trg_seq # v1
        #pdb.set_trace() 
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, query_pos,slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

'''
enc_output: from arch_feat(src_seq)->encoder
val_acc_pred: from enc_output
trg_tier_seq: from trg_seq, specific tier in trg_seq
dec_output: from trg_tier_seq and enc_output
total_logit: from dec_output, sum up the dec_output of all tiers 
'''
class TransformerQ2L(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_tier=5, n_arch_patch=19, d_patch=7,
            d_patch_vec=512, d_model=512, d_ffn_inner=2048, d_tier_prj_inner=256,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            d_val_acc_prj_inner=2048, scale_prj=True):

        super().__init__()

        self.scale_prj = scale_prj
        self.d_model = d_model
        self.n_tier = n_tier
        self.n_arch_patch = n_arch_patch
        self.d_patch = d_patch

        self.src_prj = nn.Linear(d_patch*d_patch, d_patch_vec)
        self.trg_prj = nn.Linear(d_patch_vec, d_patch_vec)

        self.encoder = Encoder(
            n_position=n_position, d_patch_vec=d_patch_vec, d_model=d_model, d_inner=d_ffn_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position, d_patch_vec=d_patch_vec, d_model=d_model, d_inner=d_ffn_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.tier_prj = nn.Sequential(
            nn.Linear(n_arch_patch * d_model, d_tier_prj_inner, bias=False),
            nn.ReLU(),
            nn.Linear(d_tier_prj_inner, n_tier)
            )

        self.val_acc_prj = nn.Sequential(
            nn.Linear(n_arch_patch * d_model, d_val_acc_prj_inner, bias=True),
            nn.ReLU(),
            nn.Linear(d_val_acc_prj_inner, 1)
            )

        #query to label
        hidden_dim=d_model
        self.query_embed = nn.Embedding(n_tier, hidden_dim) 
        self.Q2ldecoder=Q2lDecoder(
            n_position=n_position, d_patch_vec=d_patch_vec, d_model=d_model, d_inner=d_ffn_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.fc = GroupWiseLinear(n_tier, hidden_dim, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_patch_vec, 'the dimensions of all module outputs should be the same to insure residual connections'


    def forward(self, src_seq, trg_seq, return_attns = False):

        src_mask = None
        trg_mask = None
        #pdb.set_trace()
        src_seq = src_seq.view(-1, self.n_arch_patch, self.d_patch*self.d_patch) #torch.Size([256, 19, 7, 7]) ->torch.Size([256, 19, 49])

        src_seq = self.src_prj(src_seq) # linear operation: torch.Size([256, 19, 49])->torch.Size([256, 19, 256]) [batch_size,n_arch_patch,d_patch_vec]
        trg_seq = self.trg_prj(trg_seq) # linear operation: torch.Size([5, 19, 256])->torch.Size([5, 19, 256])

        enc_output, *_ = self.encoder(src_seq, src_mask)    # src_mask=None enc_output.shape=torch.Size([256, 19, 256])

        val_acc_pred = self.val_acc_prj(enc_output.clone().view(-1, self.n_arch_patch * self.d_model))  #n_arch_patch=19; d_model=256, val_acc_pred.shape=torch.Size([256, 1]) [batch_size,pred_acc]
        
        # replace
        query_input = self.query_embed.weight # query_input.shape=torch.Size([n_tier, hidden_dim])
        query_embed = query_input.unsqueeze(1).repeat(1, src_seq.size(0), 1) #torch.Size([class_num, src_seq.size(0), hidden_dim])
        query_embed=query_embed.transpose(0,1)
        #bs,n_arch_patch,w=enc_output.shape
        tgt = torch.zeros_like(query_embed)
        dec_output, *_ = self.Q2ldecoder(tgt, trg_mask, enc_output, src_mask,query_embed) #dec_output.shape=torch.Size([256, 19, 256])
        out=self.fc(dec_output) #torch.Size([256, 5])
        if self.scale_prj:
            out *= self.d_model ** -0.5 # 256**-0.5=0.0625
        return out, enc_output, val_acc_pred