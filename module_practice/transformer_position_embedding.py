import torch
import math
from torch import nn

# 这里一开始把pe变成类的属性会导致无法缓存，在 __init__ 中使用 register_buffer('pe', self.pe) 注册了 self.pe，这时 self.pe 作为一个类属性存在。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # self.pe = torch.zeros(max_len, d_model)
        self.position =  torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        pe = self._calculate_position(d_model)
        self.register_buffer('pe', pe)

    def _calculate_position(self, d_model: int) -> torch.Tensor:
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*-(math.log(10000.0)/d_model))
        pe = torch.zeros(self.position.size(0), d_model)
        pe[:, 0::2] = torch.sin(self.position*div_term)
        pe[:, 1::2] = torch.cos(self.position*div_term)
        pe = pe.unsqueeze(0)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.autograd.Variable(self.pe[:,:x.size(1)], requires_grad=False)

        return self.dropout(x)

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()  
#         self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层
        
#         # 计算位置编码并将其存储在pe张量中
#         pe = torch.zeros(max_len, d_model)                # 创建一个max_len x d_model的全零张量
#         position = torch.arange(0, max_len).unsqueeze(1)  # 生成0到max_len-1的整数序列，并添加一个维度
#         # 计算div_term，用于缩放不同位置的正弦和余弦函数
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
 
#         # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)                  # 在第一个维度添加一个维度，以便进行批处理
#         self.register_buffer('pe', pe)        # 将位置编码张量注册为缓冲区，以便在不同设备之间传输模型时保持其状态
        
#     # 定义前向传播函数
#     def forward(self, x):
#         # 将输入x与对应的位置编码相加
#         x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], 
#                          requires_grad=False)
#         # 应用dropout层并返回结果
#         return self.dropout(x)   
    
if __name__ == '__main__':
    p = PositionalEncoding(d_model=768, dropout=0.1)
    print(p.pe)


