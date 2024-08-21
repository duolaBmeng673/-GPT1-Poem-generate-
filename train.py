import torch
from torch.utils.data import Dataset, DataLoader
from Maps import word2id, id2word
from Model import *
from Dataset import *
import pandas as pd
from torch import nn
import time

torch.cuda.empty_cache()

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
    batch_size = 16

    model = PoemModel(vocab_size, num_heads, d_model, batch_size, device, num_layers).to(device)
    # 损失函数
    # 8为'<pad>'填充的索引，计算损失时忽略该值
    loss = torch.nn.CrossEntropyLoss(ignore_index = 8).to(device)
    # 使用Adam优化器
    opt = torch.optim.Adam(model.parameters(), lr)
    
    epoch =  15

    train_data = MyDataset(data, word2id, id2word)
    train_data = DataLoader(train_data, batch_size, shuffle=True, drop_last=True, collate_fn=train_data.padding_batch)

    for i in range(epoch):
        print(f"第{i}轮训练开始")
        sum_loss = 0
        # 已处理的批数
        cnt_data = 0
        start_time = time.perf_counter()
        for batch in train_data:
            opt.zero_grad()
            input,target = batch
            # input, target形状：(batch_size, max_len)
            input = input.to(device)
            target = target.to(device)
            # 将输出数据展平为一维向量
            target = target.view(-1)

            # 前向传播
            output = model.forward(input).to(device)
            # output形状(batch_size, len_s, vocab_size)
            # 在保证值不变的前提下对矩阵进行重构
            output = output.view(-1, vocab_size) # ?

            # 计算当前批次的损失值，返回标量张量
            # output：模型输出  target：实际结果
            result_loss = loss(output, target)
            # 反向传播，计算梯度
            result_loss.backward()
            # 更新模型参数，应用通过反向传播计算出的梯度
            opt.step()

            sum_loss += result_loss
            cnt_data += 1
            
        end_time = time.perf_counter()
        # item()将单个标量张量转化为数值类型
        print(f"第{i}轮的平均损失为{sum_loss.item()/cnt_data}")
        print(f"第{i}轮花费时间为：{end_time - start_time}")

    torch.save(model.state_dict(), './model_state.pth')

