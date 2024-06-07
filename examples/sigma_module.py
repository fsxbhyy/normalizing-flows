import torch
import torch.nn as nn
from funcs_sigma import *


class Sigma(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def func100(self, x):
        return func_sigma_o100.graphfunc(x)

    def func200(self, x):
        return func_sigma_o200.graphfunc(x)

    def func300(self, x):
        return func_sigma_o300.graphfunc(x)

    def func400(self, x):
        return func_sigma_o400.graphfunc(x)

    def func500(self, x):
        return func_sigma_o500.graphfunc(x)

    def func600(self, x):
        return func_sigma_o600.graphfunc(x)
