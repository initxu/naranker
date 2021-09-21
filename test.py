import torch
import json
import time
import torch.nn.functional as F



if __name__ == '__main__':

    ######### for test ranker

    # from architecture.arch_encode import feature_tensor_encoding
    # from ranker.Models import Transformer

    # data_path = '/home/ubuntu/workspace/nar/target.json'
    # with open(data_path, 'r') as f:
    #     dataset = json.load(f)
    # f.close()
    # assert isinstance(dataset, list)
    # arch = dataset[0]

    # # batch_sz build
    # b_sz = 256
    # x = feature_tensor_encoding(arch)  # 1*19*7*7 batch*19*7*7
    # b_x = torch.stack([x for _ in range(b_sz)], dim=0).cuda()
    # decoder_input = torch.randn(1, 19, 512).float().cuda()

    # ranker = Transformer(
    #     n_tier=5,
    #     n_arch_patch=19,
    #     d_patch=7,
    #     d_patch_vec=512,
    #     d_model=512,
    #     d_ffn_inner=2048,
    #     d_tier_prj_inner=4096,
    #     n_layers=6,
    #     n_head=8,
    #     d_k=64,
    #     d_v=64,
    #     dropout=0.1,
    #     n_position=200,
    #     scale_prj=False)

    # ranker.cuda()

    # l_seq_logit = []
    # s1 = time.time()
    # for i in range(5):
    #     seq_logit = ranker(b_x.float(), decoder_input)
    #     # print(seq_logit.shape)
    #     l_seq_logit.append(seq_logit)
    # s2 = time.time()
    
    # total_logit = torch.zeros(seq_logit.shape, dtype=seq_logit.dtype,device=seq_logit.device)
    # for _,te in enumerate(l_seq_logit):
    #     total_logit += te
    # prob = F.softmax(total_logit, dim=-1)

    ######### for test sampler
    # for test _sample_target_value
    # import copy
    # counts_dict = {5758991: 232, 11517982: 93, 17276973: 38, 23035964: 31, 28794955: 8, 34553946: 18, 40312937: 2, 46071928: 1}
    # counts_dict2 = {5758991: 1, 11517982: 2, 17276973:18, 23035964: 8, 28794955: 31, 34553946: 38, 40312937: 93, 46071928: 232}
    # counts_dict3 = copy.deepcopy(counts_dict)
    # counts_dict4 = copy.deepcopy(counts_dict2)
    # counts_dict5 = copy.deepcopy(counts_dict)
    # counts_dict6 = copy.deepcopy(counts_dict2)
    # list_candi1 = [counts_dict,counts_dict2]
    # list_candi2 = [counts_dict3,counts_dict4]
    # list_candi3 = [counts_dict5,counts_dict6]
    # candi_dict = {}
    # candi_dict['flops'] = list_candi1
    # candi_dict['params'] = list_candi2
    # candi_dict['n_nodes'] = list_candi3
    # sampler = ArchSampler(n_tier=1,threshold_kl_div=27,batch_size=423,batch_factor=1,reuse_step=10)
    # v = sampler.sample_subnets(candi_dict, 100, kl_factor_list=[1,1,1])
    # print(v)
    
    #  for test __sample_edges_and_types

    from sampler.ArchSampler import ArchSampler
    type_dict = {'input':1, 'conv1x1-bn-relu':2 , 'conv3x3-bn-relu':3, 'maxpool3x3':4, 'output':5}
    sampler = ArchSampler(n_tier=1,threshold_kl_div=27,batch_size=423,batch_factor=1,reuse_step=10,node_type_dict=type_dict)
    v = sampler._sample_edges_and_types(7,[7])
    print(v)
   