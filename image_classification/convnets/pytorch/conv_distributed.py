from milabench.perf import *
from milabench.perf.fp16utils import OptimizerAdapter, ModelAdapter

import os
import torch
import torch.nn as nn
import torch.distributed as distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# ----
parser = parser_base()
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR')
parser.add_argument('--half', action='store_true', default=False)
parser.add_argument('--device_ids', type=str, default=None)


# ----
exp = Experiment(__file__)
args = exp.get_arguments(parser, show=True)
node_size = torch.cuda.device_count()

if args.device_ids is not None:
   args.device_ids = list(map(lambda x: int(x), args.device_ids.split(',')))
   node_size = len(args.device_ids)

# Batch size it per GPU
args.batch_size = args.batch_size * node_size
device = exp.get_device()
chrono = exp.chrono()
try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except:
    pass


# ----
model = models.__dict__[args.arch]()
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = OptimizerAdapter(torch.optim.SGD(
    model.parameters(),
    args.lr),
    half=args.half,
    dynamic_loss_scale=True
)

# ----
model = ModelAdapter(model, half=args.half)
model = nn.DataParallel(model, device_ids=args.device_ids)


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
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
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
        for id in range(args.number):

            # data loading do not start here so naturally this is not data loading
            # only the time waiting for the data loading to finish
            with chrono.time('loading'):
                input, target = next_batch()

                input = input.cuda()
                target = target.cuda()

            with chrono.time('compute'):
                output = model(input)
                loss = criterion(output, target)

                exp.log_batch_loss(loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()

    exp.show_eta(epoch, t)

# each GPU did the same batch size
exp.report()

