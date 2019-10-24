from perf import *
from perf.fp16utils import OptimizerAdapter, ModelAdapter


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
parser.add_argument('--opt-level', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world-size", default=None, type=int)
parser.add_argument("--backend", default='gloo', type=str)

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


# ---
def printf(*vargs, **kwargs):
    if args.local_rank == 0:
        print(*vargs, **kwargs)


if args.world_size is None:
    args.world_size = torch.cuda.device_count()

printf('> Init Distributed')
torch.distributed.init_process_group(
    backend=args.backend,
    init_method='env://',
    world_size=args.world_size,
    rank=args.local_rank
)
printf(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
world_size = torch.distributed.get_world_size()
os.environ['WORLD_SIZE'] = str(world_size)
device = torch.cuda.device(args.local_rank)
printf('< Init Distributed')

# ----
model = models.__dict__[args.arch]()
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

# ----
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
sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    sampler=sampler
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


def reduce_tensor(tensor):
    rt = tensor.clone()
    # if nccl API is not avaible we cannot to the reduce on GPU
    if not torch.distributed.is_nccl_available():
        rt = rt.cpu()

    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= world_size
    return rt.cuda()


model.train()
for epoch in range(args.repeat):
    sampler.set_epoch(epoch)

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
                loss = reduce_tensor(loss)

                if args.local_rank == 0:
                    exp.log_batch_loss(loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()
                optimizer.backward()
                optimizer.step()

    if args.local_rank == 0:
        exp.show_eta(epoch, t)

if args.local_rank == 0:
    # each GPU did the same batch size
    args.batch_size *= world_size
    exp.report()
