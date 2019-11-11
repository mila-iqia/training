from __future__ import print_function
from milabench.perf import *

import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


exp = Experiment(__file__)

parser = parser_base(description='VAE MNIST Example')
args = exp.get_arguments(parser, show=True)
device = exp.get_device()
chrono = exp.chrono()


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

data_folder = os.path.join(os.environ['DATA_DIRECTORY'], 'mnist')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_folder, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_folder, train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def test(epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():

        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),  'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":

    model.train()

    for epoch in range(args.repeat):
        train_loss = 0
        start = 0

        with chrono.time('train') as t:
            while start < args.number:
                batch_id = start
                for batch_id, (data, _) in enumerate(train_loader, start=start):
                    if batch_id >= args.number:
                        break

                    data = data.to(device)

                    optimizer.zero_grad()

                    recon_batch, mu, logvar = model(data)

                    loss = loss_function(recon_batch, data, mu, logvar)

                    loss.backward()

                    exp.log_batch_loss(loss.item())

                    train_loss += loss.item()

                    optimizer.step()

                start = batch_id

        exp.log_epoch_loss(train_loss)
        exp.show_eta(epoch, t)

    exp.report()

