# 构建模型

import torch
from torch import nn
import math

print(torch.__version__)
print(torch.cuda.is_available())

# Positional Embedding位置嵌入层
class GPT1_Embedding_Layer(nn.Module):
    # 初始化函数
    # vocab_size：词汇表大小；embedding_dim：嵌入向量维度
    # 模型能处理最大序列长度暂不定义
    # dropout_rate：dorpout层比率默认为0.1
    def __init__(self, vocab_size, embedding_dim, device):
        super(GPT1_Embedding_Layer, self).__init__(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device
        # 定义嵌入层，将tokens索引映射到多维嵌入向量
        # nn.Embedding()接受两个参数：词汇表大小和嵌入向量的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # 输入矩阵x形状为(batch_size, seq_len)
    # 位置编码
    def Position_encoding(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        dim = self.embedding_dim
        PE = torch.zeros([batch_size, seq_len, dim], dtype=torch.float32).to(self.device)

        # arange输出一维张量(rows, )，包含0到seq_len-1的整数，并用reshape()扩张到二维，形状为(seq_len, 1)
        Positional_matrix = torch.arange(seq_len).reshape((-1, 1))
        # 从0开始，步长为2，张量包含所有小于dim的数，reshape()扩张二维，形状为(1, dim//2)
        exponent = torch.arange(0, dim, 2).reshape((1, -1)) / dim
        # 实现transformers位置编码并返回矩阵
        # 矩阵X形状为(seq_len, dim//2)
        X = Positional_matrix / torch.pow(10000, exponent)
        # 进行torch广播计算，最后形成的矩阵PE由batch_size个形状为(seq_len, dim)的相同矩阵组合而成
        PE[:,:,0::2] = torch.sin(X)
        PE[:,:,1::2] = torch.cos(X)
        return PE
    
    def forward(self, x):
        x = self.embedding(x)
        # 经嵌入函数输出的矩阵x形状为(batch_size, seq_len, dim)，与位置编码函数输出矩阵形状相同
        x += self.Position_encoding(x)
        return x
    
# Scaled Dot-Product Attention点积缩放注意力
class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, device):
        super(Scaled_Dot_Product_Attention, self).__init__(device)
        self.device = device
        # 使用softmax函数，dim = -1意味着softmax函数作用到随后一个维度上
        self.softmax = nn.Softmax(dim = -1)

    def mask(self, dim_1, dim_2, dim_3):
        # 建立dim_3 * dim_3上三角矩阵，对角线上方值为true，之后扩展为mask形状
        mask = torch.triu(torch.ones([dim_1, dim_2, dim_3, dim_3], dtype=torch.bool), diagonal = 1).to(self.device)
        return mask
    
    # 实现点积缩放注意力
    def forward(self, Q, K, V):
        # transpose:转置K的第二维度和最后维度
        # K.size(-1)的最后维度为d_k
        attention_scores = torch.matmul(Q, K.transpose(2, -1)) / math.aqrt(K.size(-1))
        sub_mask = self.mask(attention_scores.size(0), attention_scores.size(1), attention_scores.size(2))
        # 将sub_mask矩阵值为True的单位转换成-1e7
        attention_scores.masked_fill_(sub_mask, -1e7)
        # 计算softmax
        attention_scores = self.sf(attention_scores)
        # score形状(batch_size, num_heads, seq_len, d_v)
        score = torch.matmul(attention_scores, V)
        return score
        
class Multi_Head_Attn(nn.Module):
    def __init__(self, num_heads, d_model, batch_size, device):
        super(Multi_Head_Attn, self).__init__(num_heads, d_model, batch_size, device)
        self.d_k = self.d_v = self.d_q = 64
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.d_model = d_model
        self.device = device
        # nn.Linear函数功能：将输入特征变换为输出特征
        self.w_Qs = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_Ks = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_Vs = nn.Linear(d_model, num_heads * self.d_v, bias=False)
        self.w_Os = nn.Linear(num_heads * self.d_v, d_model, bias=False)
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
