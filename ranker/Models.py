''' Define the Transformer model '''
import torch
import torch.nn as nn
from ranker.Layers import EncoderLayer, DecoderLayer
from ranker.Modules import PositionalEncoding
from ranker.Modules import get_pad_mask,get_subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)      # n_src_vocab=9521, d_word_vec=512, padding_idx表明输入的序列的第pad_idx位的嵌入不贡献梯度，即pad_idx位的嵌入训练时不会更新
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([                                                  # 堆叠n_layers=6次[多头注意力+前馈层]
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)                                             # (256,36,512), 36个词，每个词512的编码
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5                                               # 对词编码的输出乘上/sqrt(d_model)
        enc_output = self.dropout(self.position_enc(enc_output))                            # 对缩放后的词编码，加上位置编码，dropout       !!!!!!!!!!这里是不是要加dropout
        enc_output = self.layer_norm(enc_output)                                            # 层归一化                                  !!!!!!!!!!这里是不是要加层归一化

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)               # 堆叠n_layers=6次 [多头注意力w/mask+多头注意力+前向层]
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))                            # ！！！！！！！！！！是否要加dropout
        dec_output = self.layer_norm(dec_output)                                            # ！！！！！！！！！！是否要加层归一化

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
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False            # note: scale_emb不是实例变量，是个暂存的值，用于初始化encoder和decoder
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False       # scale_prj是个示例变量，用于forward时乘线性层的输出
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight
        """
        trg_word_emb和src_word_emb分别是decoder和encode入口处的nn.embedding
        trg_word_prj是最后输出的Linear层
        """


    def forward(self, src_seq, trg_seq):
        """
        src_seq:输入序列,(256,36)
        src_mask:输入序列的mask,(256,1,36)
        trg_seq:目标序列，(256,32)
        trg_mask:目标序列的mask, (256,32,32)
        """
        # encoder的mask 可能要考虑矩阵不足7的pad的列
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)                                      # mask是对输入序列，不等于1（词典中blank对应的）对应的位置为True，等于1的位置是False的矩阵，同时在倒数第二维加一个维度
        # 目标的mask也要考虑pad掉不满足7的列节点的数据，也可能因为这里的编码已经是提取过的，完全去掉这个mask
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)       # 前一个是针对空格的mask(256,1,32), 后一个是针对后面序列的mask(1,32,32), 两个求与,得到目标序列的mask(256,32,32)

        enc_output, *_ = self.encoder(src_seq, src_mask)                                        # enc_output(256,36,512)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)                  # dec_output(256,32,512)
        seq_logit = self.trg_word_prj(dec_output)                                               # seq_logit(256, 32, 9521), 目标序列32个词汇，每个词对应的字典中词的概率，字典的长度是9521
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
