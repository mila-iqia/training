#!/usr/bin/env python
from __future__ import print_function
from perf import *
from itertools import count

import torch
import torch.nn.functional as F
import torch.nn as nn

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y



# add benchmark arguments
parser = parser_base()
exp = Experiment(__file__)
args = exp.get_arguments(parser, show=True)
device = exp.get_device()
chrono = exp.chrono()


# Define model
fc = torch.nn.Linear(W_target.size(0), 1)
fc.to(device)

for epoch in range(args.repeat):

    with chrono.time('train') as t:
        for batch_idx in range(args.number):
            # Get data
            batch_x, batch_y = get_batch(args.batch_size)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Reset gradients
            fc.zero_grad()

            # Forward pass
            output = F.smooth_l1_loss(fc(batch_x), batch_y)
            loss = output.item()

            exp.log_batch_loss(loss)

            # Backward pass
            output.backward()

            # Apply gradients
            for param in fc.parameters():
                param.data.add_(-0.01 * param.grad.data)

    exp.show_eta(epoch, t)

print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))

exp.report()
