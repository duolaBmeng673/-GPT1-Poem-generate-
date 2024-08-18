# 构建模型

import torch
from torch import nn

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
        X = Positional_matrix / torch.pow(10000, exponent)
        PE[:,:,0::2] = torch.sin(X)
        PE[:,:,1::2] = torch.cos(X)
        return PE
    
    def forward(self, x):
        x = self.embedding(x)
        x += self.Position_encoding(x)
        return x
    
# Scaled Dot-Product Attention点积缩放注意力
class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, device):
        super(Scaled_Dot_Product_Attention, self).__init__(device)
        self.device = device
        self.softmax = nn.Softmax(dim = -1)

    
