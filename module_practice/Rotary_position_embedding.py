import torch
from torch import nn
from transformers import LlamaModel, T5Model

class LlmamRotaryEmbedding(nn.Module):
    """这个只是制造sin cos，是对每个位置进行信息提取，mod,当前进制下当前层级的模， 每个位置都会由最大位置的维度去提取，也就是说， 当位置为1时，依旧这个
    位置1会有一个维度为512的提取向量公式，在这个向量中，每个元素都是一个公式 mod公式。
    """
    def __init__(self, dim, max_position_embedding=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim 
        self.max_position_embeddings = max_position_embedding
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)/self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embedding

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None:, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 是对每个位置进行信息提取，mod,当前进制下当前层级的模
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # 这里有个小技巧 就是直接拼接两个256的维度，前一个
            emb = torch.cat([freqs, freqs], dim=-1)
            # emb = emb[...,:].repeat_interleave(2, dim=-1) 论文写法
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaLinearScalingRotaryEmbedding(LlmamRotaryEmbedding):
    def forward(self, x, postion_ids):
        postion_ids = postion_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, postion_ids)
        return cos, sin 

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., :x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, postion_ids=None, unsqueese_dim=1):
    cos = cos.unsqueeze(unsqueese_dim)
    sin = sin.unsqueeze(unsqueese_dim)
    # 这里跟论文中的公式不一样
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 论文写法 结合上面那个
    # q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    # q2 = q2.reshape(q.shape)    

    # k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    # k2 = k2.reshape(k.shape)
    # q_embed = (q2 * cos) + (q2 * sin)
    # k_embed = (k2 * cos) + (k2 * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, postion_ids=None, unsqueese_dim=1):
    cos = cos.unsqueeze(unsqueese_dim)
    sin = sin.unsqueeze(unsqueese_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# paper formula


    
    