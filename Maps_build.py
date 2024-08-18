# 功能：建立word2id、id2word映射字典
# 结果已经保存到Maps.py中，除非更改训练数据集，否则无需重复运行

import pandas as pd


def Maps_Build(data):
    with open('Maps.py', 'w') as file:

        dict_list = ['<BOS>','<EOS>','<UNK>','题目','关键词','内容',',',':','<pad>','五言','七言',' ']
        dict_list_1 = dict_list
        plus_dic = len(dict_list_1)

        for i in range(0, len(data)):
            title = str(data.iloc[i]['title'])
            content = str(data.iloc[i]['content'])
            keywords = str(data.iloc[i]['keywords'])

            # 字符串合并,同时删除空格
            input_string = "".join((title + content + keywords).split())
            for char in input_string:
                dict_list.append(char)

        # 将dict_list转化为集合,再转化为列表,转化为集合后自动去重
        # 为方便后续写代码，将dict_list_1中的元素放在字典最前方
        dict_list = list(set(dict_list[plus_dic:]))
        id2word = {}
        word2id = {}
        for i in range(0, (len(dict_list) + plus_dic)):
            if i < plus_dic:
                id2word[i] = dict_list_1[i]
                word2id[dict_list_1[i]] = i
            else:
                id2word[i] = dict_list[i - plus_dic]
                word2id[dict_list[i - plus_dic]] = i
        id2word_str = "id2word = " + str(id2word)
        word2id_str = "word2id = " + str(word2id)
        file.write(id2word_str + '\n')
        file.write(word2id_str)


train_data = pd.read_json('./data_test/ccpc_train_v1.0.json', lines=True)
Maps_Build(train_data)