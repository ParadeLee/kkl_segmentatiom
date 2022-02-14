import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LePEAttention(nn.Module):
     def __init__(self, dim, resolution, idx, split_size=7, ):