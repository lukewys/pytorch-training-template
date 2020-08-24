import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyModel(nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.model_type = 'dummy_model'

    def forward(self, x):
        return self.fc(x)
