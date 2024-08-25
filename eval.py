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
    # model.load_state_dict(torch.load('model_state.pth'))
    state_dict = torch.load('model_state.pth', weights_only=True)
    model.load_state_dict(state_dict)

    # reduction = 'mean' 对所有样本的损失取平均值，也就是将每个样本的损失相加，然后除以样本的数量
    loss = torch.nn.CrossEntropyLoss(ignore_index = 8,reduction = 'mean')
    loss.to(device)

    raw_val_data = pd.read_json('./data_test/ccpc_valid_v1.0.json', lines=True)
    val_data = Mytest_Dataset(raw_val_data, word2id, id2word)
    val_data = DataLoader(val_data, batch_size, shuffle=True, drop_last=True, collate_fn=val_data.padding_batch)

    sum_loss = 0
    lenthth = 0
    softmax = nn.Softmax(dim = -1)
    with torch.no_grad():
        for batch in val_data:
            input, target, lenth = batch
            input = input.to(device)
            target = target.to(device)
            # print(f"target shape:{target.shape}")
            target = target.view(-1)
            # print(f"target:{re_tokenizer_1(target, id2word)}")

            re_tokenizer(input.view(-1), id2word)

            output_sen = ''
            output_logit = torch.zeros([lenth, vocab_size],dtype=torch.float32,device = device)
            for i in range(lenth):
                result = model.forward(input)
                next_token = result[0, result.size(1) - 1,:]
                output_logit[i,:] = next_token
                next_logit = softmax(next_token)
                token = torch.argmax(next_logit, dim = -1)
                output_sen += id2word[token.item()]
                input = torch.cat((input, token.unsqueeze(0).unsqueeze(0)), dim = 1)
                
            # print(f"len of output_sen:{len(output_sen)}")
            # print(f"shape of output logit:{output_logit.shape}")
            # print(f"shape of target:{target.shape}")
            # print(output_logit)
            # print(target)
            poem_loss = loss(output_logit, target)
            print(f"{output_sen},ppl:{torch.exp(poem_loss)}\n")
            sum_loss += poem_loss
            lenthth += 1
    mean_loss = sum_loss / lenthth
    print(f"在整个数据集上的平均PPL为{torch.exp(mean_loss)}")