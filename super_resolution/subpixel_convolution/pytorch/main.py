from __future__ import print_function
from perf import *

import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set


# Training settings
parser = parser_base(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--no-checks', action='store_true', default=False, help='do not save checkpoints')


exp = Experiment(__file__)
opt = exp.get_arguments(parser, show=True)
device = exp.get_device()

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.workers, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.workers, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
model.train()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    start =0
    while start < opt.number:
        batch_id = start

        for batch_id, batch in enumerate(training_data_loader, start=start):
            if batch_id >= opt.number:
                break

            input, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            loss = criterion(model(input), target)
            exp.log_batch_loss(loss.item())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        start = batch_id

    return epoch_loss


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr

    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):
    if opt.no_checks:
        return

    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


chrono = exp.chrono()

for epoch in range(opt.repeat):
    epoch_loss = 0

    with chrono.time('train') as t:
        exp.log_epoch_loss(train(epoch))

    psnr = test()

    exp.log_metric('psnr', psnr)
    exp.show_eta(epoch, t, f'PSNR: {psnr}')

exp.report()
