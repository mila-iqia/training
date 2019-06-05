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
parser.add_argument('--loader-batch-size', type=int, default=None)

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
if args.loader_batch_size is None:
    args.loader_batch_size = args.batch_size


# Not really improving loading speed
class BatchChunksIterator:
    """ Pytorch dataloader works better with big batch sizes.
        so we allow the user to specify a bigger batch size for the dataloader only.
        but the input size will be smaller.

        [In] < ./image_classification/convnets/pytorch/run.sh /Tmp/mlperf/data//data/ImageNet/train/ --repeat 20 --number 5 --workers 8 --arch resnet18 --cuda --opt-level O0 --batch-size 64 --loader-batch-size 64

        [Out]>   "avg": 384.77666779370503 img/sec

    """

    def __init__(self, dataloader: DataLoader, batch_size_input,):
        self.loader = iter(dataloader)
        self.bs_loader = dataloader.batch_size
        self.bs_input = batch_size_input
        self.disabled = False

        # The loader should have a bigger batch size for this to matter
        if self.bs_loader <= self.bs_input:
            self.disabled = True
            print('/!\\ BatchChunkIterator is disabled')

        self.bs_count = self.bs_loader // self.bs_input
        assert self.bs_loader % self.bs_input == 0

        self.current_input = None
        self.current_target = None
        self.i = self.bs_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.disabled:
            return next(self.loader)

        if self.i == self.bs_count:
            self.i = 0
            self.current_input, self.current_target = next(self.loader)

        start = self.i * self.bs_input
        end = start + self.bs_input
        self.i += 1

        return self.current_input[start:end, :], self.current_target[start:end]


train_loader = DataLoader(
    train_dataset,
    batch_size=args.loader_batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True
)

# dataset is reduced but should be big enough for benchmark!
batch_iter = BatchChunksIterator(train_loader, args.batch_size)


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
