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
model = nn.Linear(5, 10, bias=False)
print(model(X))


