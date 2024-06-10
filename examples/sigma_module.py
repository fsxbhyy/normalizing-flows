import torch
import torch.nn as nn
from funcs_sigma import *


class Sigma(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def func100(self, y, x):
        eval_graph100(y, x)

    def func200(self, y, x):
        eval_graph200(y, x)

    def func300(self, y, x):
        eval_graph300(y, x)

    def func400(self, y, x):
        eval_graph400(y, x)

    def func500(self, y, x):
        eval_graph500(y, x)

    def func600(self, y, x):
        eval_graph600(y, x)
