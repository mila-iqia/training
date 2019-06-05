from perf import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from apex import amp
import time

# ----
parser = parser_base()
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
parser.add_argument('--opt-level', type=str)

# ----
exp = Experiment(__file__)
args = exp.get_arguments(parser, show=True)
device = exp.get_device()
chrono = exp.chrono()
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except:
    pass


# ----
model = models.__dict__[args.arch]()
model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    args.lr)

# ----
model, optimizer = amp.initialize(
    model,
    optimizer,
    enabled=args.opt_level != 'O0',
    cast_model_type=None,
    patch_torch_functions=True,
    keep_batchnorm_fp32=None,
    master_weights=None,
    loss_scale="dynamic",
    opt_level=args.opt_level
)


# ----
train_dataset = datasets.ImageFolder(
    args.data,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
)

# ----
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True
)

# dataset is reduced but should be big enough for benchmark!
batch_iter = iter(train_loader)


def next_batch():
    global batch_iter
    try:
        return next(batch_iter)
    except StopIteration:
        batch_iter = iter(train_loader)
        return next(batch_iter)


model.train()
for epoch in range(args.repeat):

    with chrono.time('train') as t:
        s = time.time()
        for id in range(args.number):

            # data loading do not start here so naturally this is not data loading
            # only the time waiting for the data loading to finish
            with chrono.time('loading'):
                input, target = next_batch()

                input = input.to(device)
                target = target.to(device)

            with chrono.time('compute'):
                output = model(input)
                loss = criterion(output, target)

                exp.log_batch_loss(loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

    exp.show_eta(epoch, t)

exp.report()
