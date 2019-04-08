from __future__ import print_function
from perf import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time


folder = os.environ['DATA_DIRECTORY']
data_folder = folder + '/time_series_prediction'


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

        self.device = next(self.parameters()).device
        self.dtype = next(self.parameters()).dtype

    def to(self, device=None, dtype=None):
        model = super().to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return model

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)
        c_t = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)
        h_t2 = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)
        c_t2 = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


to_type = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}


if __name__ == '__main__':
    parser = parser_base()
    parser.add_argument('--dtype', type=str, help='model float type', default='float32')

    exp = Experiment(__file__)
    args = exp.get_arguments(parser, show=True)
    device = exp.get_device()
    dtype = to_type[args.dtype]
    chrono = exp.chrono()

    # load data and make training set
    data = torch.load(data_folder + '/traindata.pt')

    input = torch.from_numpy(data[3:, :-1]).to(device=device, dtype=dtype)
    target = torch.from_numpy(data[3:, 1:]).to(device=device, dtype=dtype)

    test_input = torch.from_numpy(data[:3, :-1]).to(device=device, dtype=dtype)
    test_target = torch.from_numpy(data[:3, 1:]).to(device=device, dtype=dtype)

    # build the model
    seq = Sequence().to(device=device, dtype=dtype)
    criterion = nn.MSELoss().to(device=device, dtype=dtype)

    optimizer = optim.SGD(seq.parameters(), lr=0.01)

    total_time = 0

    seq.train()
    # begin to train
    for i in range(args.repeat):
        with chrono.time('train') as t:
            for _ in range(args.number):

                def closure():
                    optimizer.zero_grad()
                    out = seq(input.to(device=device, dtype=dtype))
                    loss = criterion(out, target)
                    loss.backward()
                    exp.log_batch_loss(loss.item())
                    return loss

                optimizer.step(closure)
        exp.show_eta(i, t)

    exp.report()
