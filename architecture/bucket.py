""""
Data: 2021/09/15
Target: Store and Update tier Embedding
Method: element-wise add all the embeddings belong to this tier and devided by the number of embeddings for scaling
"""

import torch

class Bucket(object):

    n_tier = 0  # 类变量，每实例化一个类计数加一

    def __init__(self, flag_tier, name_tier, n_arch_patch=19, d_patch_vec=512):
        
        assert flag_tier == Bucket.n_tier, "tier flag should be the same with the number of the Bucket instances"
        self.flag_tier = flag_tier
        self.name_tier = name_tier
        self._n_arch_patch = n_arch_patch
        self._d_patch_vec = d_patch_vec

        self._total_bucket_emb = torch.zeros(
            n_arch_patch, d_patch_vec).unsqueeze(dim=0)         # [1,19,512]
        self.current_bucket_emb = torch.zeros(
            n_arch_patch, d_patch_vec).unsqueeze(dim=0)         # [1,19,512]

        self._emb_count = 0

        Bucket.n_tier += 1

    def get_bucket_emb(self):
        return self.current_bucket_emb

    def updata_bucket_emb(self, input_emb):     # input_emb[n, 19, 512]
        assert self._n_arch_patch == input_emb.size(1), "Wrong patch length"
        assert self._d_patch_vec == input_emb.size(2), "Wrong patch embedding dimension"
        n_input_emb = input_emb.size(0)

        if input_emb.is_cuda:
            self._total_bucket_emb = self._total_bucket_emb.cuda(input_emb.device)

        added_input_emb = input_emb.sum(dim=0, keepdim=True)  # [n,19,512] → [1,19,512]
        self._total_bucket_emb += added_input_emb                      # 存储的是历史上全部编码的综合
        self._emb_count += n_input_emb                          # 更新tier总数

        self.current_bucket_emb = self._total_bucket_emb / self._emb_count

        assert self.current_bucket_emb.size(0) == 1, "The length of bucket emb dimension should be 1"

        return self.current_bucket_emb

    @property
    def emb_count(self):            # emb_count不允许外界改变, 仅能通过调用此getter获得数值，方法：instance_name.emb_count
        return self._emb_count        

    @classmethod
    def get_n_tier(cls):
        return cls.n_tier

    @classmethod
    def reset_n_tier(cls):
        cls.n_tier = 0

    def __del__(self):
        Bucket.n_tier -= 1


if __name__ == '__main__':

    # for input dimension debug
    # import torch.nn.functional as F
    # logit = torch.randn(256, 5)
    # prob = F.softmax(logit, dim=-1)
    # _, tier_index = prob.max(dim=-1)
    # bucket_emb = torch.randn(256, 19, 512)
    # for i in range(5):
    #     tier_mask = (i == tier_index)
    #     tier_emb = bucket_emb[tier_mask]
    #     print('tier{} emb shape is {}'.format(i, tier_emb.shape))
    # """
    # tier0 emb shape is torch.Size([51, 19, 512])
    # tier1 emb shape is torch.Size([50, 19, 512])
    # tier2 emb shape is torch.Size([47, 19, 512])
    # tier3 emb shape is torch.Size([67, 19, 512])
    # tier4 emb shape is torch.Size([41, 19, 512])
    # """

    # for function and instantiate debug
    t = Bucket(0,'tier1',19,512)
    emb = torch.randn(40,19,512).cuda()
    t.updata_bucket_emb(emb)
    t2 = t.get_bucket_emb()
    print(t2.grad)
