import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from Model import PoemModel
from Maps import id2word, word2id
from Dataset import *

if __name__ == '__main__':
    device = torch.device("cuda")

    vocab_size = len(word2id)
    d_model = 768
    num_heads = 8
    num_layers = 12
    batch_size = 1

    model = PoemModel(vocab_size, num_heads, d_model, batch_size, device, num_layers).to(device)
    model.load_state_dict(torch.load('model_state.pth'))

    # reduction = 'mean' 对所有样本的损失取平均值，也就是将每个样本的损失相加，然后除以样本的数量
    loss = torch.nn.CrossEntropyLoss(ignore_index = 8,reduction = 'mean')
    loss.to(device)

    raw_val_data = pd.read_json('./data_test/ccpc_valid_v1.0.json', lines=True)
    val_data = Mytest_Dataset(raw_val_data, word2id, id2word)
    val_data = DataLoader(val_data, batch_size, shuffle=True, drop_last=True, collate_fn=val_data.padding_batch)

    sum_loss = 0
    lenth = 0
    softmax = nn.Softmax(dim = -1)
    with torch.no_grad():
        for batch in val_data:
            input, target, len = batch
            input = input.to(device)
            target = target.to(device)
            target = target.view(-1) 

            output_sen = ''
            output_logit = torch.zeros([len, vocab_size],dtype=torch.float32,device = device)
            for i in range(len):
                result = model.forward(input)
                next_token = result[0, result.size(1) - 1,:]
                output_logit[i,:] = next_token

                next_logit = softmax(next_token)
                token = torch.argmax(next_logit, dim = -1)
                output_sen += id2word[token.item()]
                inp = torch.cat((input, token.unsqueeze(0).unsqueeze(0)), dim = 1)

            poem_loss = loss(output_logit, target)
            print(f"{output_sen},ppl:{torch.exp(poem_loss)}\n")
            sum_loss += poem_loss
            lenth += 1
    mean_loss = sum_loss / lenth
    print(f"在整个数据集上的平均PPL为{torch.exp(mean_loss)}")