''' Define the Transformer model '''
import torch.nn as nn
from ranker.Layers import EncoderLayer, DecoderLayer
from ranker.Modules import PositionalEncoding

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_patch_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1):

        super().__init__()

        self.position_enc = PositionalEncoding(d_patch_vec, n_position=n_position)
        # self.dropout = nn.Dropout(p=dropout)                                                # 删掉，可以做ablation exp
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        # src_seq = self.dropout(src_seq)                                                   # 删掉，后期做ablation exp
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
        # self.dropout = nn.Dropout(p=dropout)                                                # 去掉，做ablation exp
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        # trg_seq = self.dropout(trg_seq)                                                   # 去掉，做ablation exp
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


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_tier=5, n_arch_patch=19, d_patch=7,
            d_patch_vec=512, d_model=512, d_ffn_inner=2048, d_tier_prj_inner=4096,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            scale_prj=True):

        super().__init__()

        self.scale_prj = scale_prj          # 用于对最后的Linear层做输出缩放
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
            nn.Linear(d_tier_prj_inner, n_tier)
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_patch_vec, 'the dimensions of all module outputs should be the same to insure residual connections'


    def forward(self, src_seq, trg_seq, return_attns = False):
        """
        src_seq:输入序列,(256,19)
        src_mask:输入序列的mask,None
        trg_seq:目标序列，(1,19,512)
        trg_mask:目标序列的mask, None
        """
        src_mask = None
        trg_mask = None

        src_seq = src_seq.view(-1, self.n_arch_patch, self.d_patch*self.d_patch)                # [256,19,7,7] → [256,19,49]

        src_seq = self.src_prj(src_seq)                                                         # [256,19,49] → [256,19,512]
        trg_seq = self.trg_prj(trg_seq)                                                         # [1,19,512] → [1,19,512]

        enc_output, *_ = self.encoder(src_seq, src_mask)                                        # enc_output(256,19,512)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)                  # dec_output(256,19,512)
        
        dec_output = dec_output.view(-1, self.n_arch_patch * self.d_model)                      # [256,19,512] → [256,9728]
        seq_logit = self.tier_prj(dec_output)                                                  # [256,9728] linear → [256,4096] linear → [256,5], target is 5 tier
        
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit