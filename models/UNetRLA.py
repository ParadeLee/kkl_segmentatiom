import torch
import torch as nn
import torch.nn.functional as F
import math

class RLACell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=False, cell_type='LSTM'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        self.cell_type = cell_type

        self.lstm_catconv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                      out_channels=4 * self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # cur_state is a tuple
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.lstm_catconv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class UNetRLA(nn.Module):
    def __init__(self, input_channel, n_class, kernel_size, expansion=4, bias=True):
        super().__init__()
