# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:56:32 2022

@author: Chovatiya
"""

from torch import nn
import torch
num_layers = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = nn.LSTM(input_size, self.hidden_layers, num_layers)
        self.linear = nn.Linear(self.hidden_layers, output_size)
        
    def forward(self, y):
        h_0 = torch.zeros(num_layers, 1, self.hidden_layers, dtype=torch.float32).to(DEVICE)
        c_0 = torch.zeros(num_layers, 1, self.hidden_layers, dtype=torch.float32).to(DEVICE)
        
        output, (h_0, c_0) = self.lstm1(y.view(len(y), 1, -1), (h_0, c_0))
        output = self.linear(output.view(len(y), -1))
        return output[-1]


# class LSTM(nn.Module):
#     def __init__(self, hidden_layers):
#         super(LSTM, self).__init__()
#         self.hidden_layers = hidden_layers
#         self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
#         self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
#         self.linear = nn.Linear(self.hidden_layers, 1)
        
#     def forward(self, y, future_preds=0):
#         outputs, num_samples = [], y.size(0)
#         h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
#         c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
#         h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
#         c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
        
#         for time_step in y.split(1, dim=1):
#             # N, 1
#             h_t, c_t = self.lstm1(time_step, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = self.linear(h_t2)
#             outputs.append(output)
            
#         for i in range(future_preds):
#             h_t, c_t = self.lstm1(output, (h_t, c_t))
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
#             output = self.linear(h_t2)
#             outputs.append(output)    
#         outputs = torch.cat(outputs, dim=1)
#         return outputs