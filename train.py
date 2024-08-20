import torch
from torch.utils.data import Dataset, DataLoader
from Maps import word2id, id2word
from Model import *
from Dataset import *
import pandas as pd
from torch import nn



if __name__ == '__main__':
    data = pd.read_json("./data_test/ccpc_train_v1.0.json", lines=True)

    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device("cuda")

    vocab_size = len(word2id)
    d_model = 768
    num_heads = 8
    num_layers = 12
    lr = 0.0001 # 优化器学习率
    batch_size = 32

    model = PoemModel(vocab_size, num_heads, d_model, batch_size, device, num_layers).to(device)
    # 损失函数
    # 8为'<pad>'填充的索引，计算损失时忽略该值
    loss = torch.nn.CrossEntropyLoss(ignore_index = 8).to(device)
    # 使用Adam优化器
    opt = torch.optim.Adam(model.parameters(), lr)
    epoch =  30

    train_data = MyDataset(data, word2id, id2word)
    train_data = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, collate_fn=train_data.padding_batch)
