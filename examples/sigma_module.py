import torch
import torch.nn as nn
from funcs_sigma import *


class Sigma(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def func100(self, x):
        return eval_graph100(x)

    def func200(self, x):
        return eval_graph200(x)

    def func300(self, x):
        return eval_graph300(x)

    def func400(self, x):
        return eval_graph400(x)

    def func500(self, x):
        return eval_graph500(x)

    def func600(self, x):
        return eval_graph600(x)
