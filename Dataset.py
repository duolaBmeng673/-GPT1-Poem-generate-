# 实现seq2seq字符串到序列转化，并将数据封装成Dataloader参数类型

import torch
from torch.utils.data import Dataset
from Maps import word2id


def tokenizer(sentence):
    seq_w2i = [0] * len(sentence)
    i = 0
    for word in sentence:
        if word in word2id:
            seq_w2i[i] = word2id[word]
        else:
            seq_w2i[i] = word2id['<UNK>']
        i += 1
    return seq_w2i

def Train_tokenizer(sentence):
    Ip_seq = []
    Tg_seq = []

    Ip_seq.append(0) # 开头
    Ip_seq.append(3) # 题目
    Ip_seq.append(7) # ：
    Ip_seq += tokenizer(sentence['title'])
    Ip_seq.append(6) # ,
    Ip_seq.append(4) # 关键词
    Ip_seq.append(7) # ：
    Ip_seq.append(9) if len(sentence['content']) == 23 else Ip_seq.append(10) # 将五言/七言添加到keywords中
    Ip_seq.append(11) # 空格
    Ip_seq += (tokenizer(sentence['keywords']))
    Ip_seq.append(6) # ,
    Ip_seq.append(5) # 内容
    Ip_seq.append(7) # ：

    # 将target序列填充等同数量的<pad>
    # 不算开头'<BOS>'位
    for i in range(0, len(Ip_seq) - 1):
        Tg_seq.append(8)
    Ip_seq += (tokenizer(sentence['content']))

    # 在seq2seq模型中，通常需要目标序列与输入序列对齐，以便模型可以学习如何从输入生成目标
    # 将内容写入目标序列，而其他部分用填充，以保持序列长度的一致性
    Tg_seq += (tokenizer(sentence['content']))

    # 在target序列结尾添加结束标记<EOS>
    Tg_seq.append(1)
    return Ip_seq, Tg_seq

# 代码由Train_tokenizer复制而来，不要求target与Input序列对齐
# target内只保留content内容
def Generate_tokenizer(sentence):
    Ip_seq = []
    Tg_seq = []

    Ip_seq.append(0) # 开头
    Ip_seq.append(3) # 题目
    Ip_seq.append(7) # ：
    Ip_seq += tokenizer(sentence['title'])
    Ip_seq.append(6) # ,
    Ip_seq.append(4) # 关键词
    Ip_seq.append(7) # ：
    len_content = len(sentence['content'])
    Ip_seq.append(9) if len_content == 23 else Ip_seq.append(10)
    Ip_seq.append(11) # 空格
    Ip_seq += (tokenizer(sentence['keywords']))
    Ip_seq.append(6) # ,
    Ip_seq.append(5) # 内容
    Ip_seq.append(7) # ：

    Tg_seq += (tokenizer(sentence['content']))

    # 返回值为两序列以及正文长度
    return Ip_seq, Tg_seq, len_content

class MyDataset(Dataset):
    # 初始化数据
    def __init__(self, data, word2id, id2word):
        self.data = data
        self.word2id = word2id
        self.id2word = id2word

    # __getitem__允许通过索引来访问数据集中的单个样本
    # 构造待处理的数据，格式为字典
    def __getitem__(self, index):
        Ip, Tg = Train_tokenizer(self.data.loc[index])
        Ip_len = len(Ip)
        Tg_len = len(Tg)
        return {'input':Ip, 'target':Tg, 'input_len':Ip_len, 'target_len':Tg_len}
    
    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.data)

    # padding_batch是一个辅助方法，用于处理一个批次的数据
    def padding_batch(self, batch):
        Ip_lens = [d['input_len'] for d in batch]
        Tg_lens = [d['input_len'] for d in batch]

        input_max_len = max(Ip_lens)
        target_max_len = max(Tg_lens)

        for d in batch:
            d['input'].extend([8] * (input_max_len - d['input_len']))
            d['target'].extend([8] * (target_max_len - d['target_len']))

        # 构造张量，结构为batch_size * max_len
        inputs = torch.tensor([d['input'] for d in batch], dtype=torch.long)
        outputs = torch.tensor([d['target'] for d in batch], dtype=torch.long)
        return inputs, outputs
    
# 用于处理测试集，返回值多一个content_max_len，为模型输出提供格式
class Mytest_Dataset(Dataset):
    def __init__(self, data, word2id, id2word):
        self.data = data
        self.word2id = word2id
        self.id2word = id2word
    
    def __getitem__(self, index):
        Ip, Tg, len_content= Generate_tokenizer(self.data.loc[index])
        Ip_len = len(Ip)
        Tg_len = len(Tg)
        return {'input':Ip, 'target':Tg, 'input_len':Ip_len, 'target_len':Tg_len, 'len_content': len_content}
    
    def __len__(self):
        return len(self.data)
    
    def padding_batch(self, batch):
        Ip_lens = [d['input_len'] for d in batch]
        Tg_lens = [d['input_len'] for d in batch]
        Con_lens = [d['len_content'] for d in batch]

        input_max_len = max(Ip_lens)
        target_max_len = max(Tg_lens)
        Con_max_len = max(Con_lens)

        for d in batch:
            d['input'].extend([8] * (input_max_len - d['input_len']))
            d['target'].extend([8] * (target_max_len - d['target_len']))

        inputs = torch.tensor([d['input'] for d in batch], dtype=torch.long)
        outputs = torch.tensor([d['target'] for d in batch],dtype=torch.long)
        return inputs, outputs, Con_max_len
    
def re_tokenizer(sentence, id2word):
    str = ''
    for i in range(len(sentence)):
        if i != 0:
            str += id2word[sentence[i].item()]
    print(str)

def re_tokenizer_1(sentence, id2word):
    str = ''
    for i in range(len(sentence)):    
        str += id2word[sentence[i].item()]
    print(str)