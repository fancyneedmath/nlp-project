from torch import Tensor

def shift(qkv: Tensor, bsz: int, q_len: int, group_size: int, num_heads: int, head_dim: int):
    qkv[:,num_heads//2:] = qkv[:,num_heads//2:].roll(-group_size//2,dims=2)
    qkv = qkv.transpose(1,2).reshape(bsz*(q_len//group_size),group_size//2,num_heads,head_dim).transpose(1,2)
    return qkv

if __name__ == '__main__':
    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    pass