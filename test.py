import torch
import json
import time
import torch.nn.functional as F
from arch_encoding import feature_tensor_encoding
from ranker.Models import Transformer

if __name__ == '__main__':

    data_path = '/home/ubuntu/workspace/nar/target.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    f.close()
    assert isinstance(dataset, list)
    arch = dataset[0]

    # batch_sz build
    b_sz = 256
    x = feature_tensor_encoding(arch)  # 1*19*7*7 batch*19*7*7
    b_x = torch.stack([x for _ in range(b_sz)], dim=0).cuda()
    decoder_input = torch.randn(1, 19, 512).float().cuda()

    ranker = Transformer(
        n_tier=5,
        n_arch_patch=19,
        d_patch=7,
        d_patch_vec=512,
        d_model=512,
        d_ffn_inner=2048,
        d_tier_prj_inner=4096,
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        n_position=200,
        scale_prj=False)

    ranker.cuda()

    l_seq_logit = []
    s1 = time.time()
    for i in range(5):
        seq_logit = ranker(b_x.float(), decoder_input)
        # print(seq_logit.shape)
        l_seq_logit.append(seq_logit)
    s2 = time.time()
    
    total_logit = torch.zeros(seq_logit.shape, dtype=seq_logit.dtype,device=seq_logit.device)
    for _,te in enumerate(l_seq_logit):
        total_logit += te
    prob = F.softmax(total_logit, dim=-1)
   