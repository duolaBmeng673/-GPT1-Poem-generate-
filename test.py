import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from Maps import word2id, id2word
import math

X = torch.Tensor([
        [0.1,0.2,0.3,0.3,0.3],
        [0.4,0.5,0.6,0.6,0.6],
        [0.7,0.8,0.9,0.9,0.9],
        ])
print(X)
model_1 = nn.Linear(5, 10 * 3, bias=False)
model_2 = nn.Linear(5, 10 * 3, bias=False)


print(model_1(X))
print(model_2(X))
X_1 = X.view(-1, 3)
print(X_1)



