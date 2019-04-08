from perf import *

import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


parser = parser_base('')
parser.add_argument('--data', type=str, default='/tmp/train', help='output directory')
exp = Experiment(__file__)
args = exp.get_arguments(parser, show=True)
device = exp.get_device()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

train_dataset = datasets.ImageFolder(
    args.data,
    data_transforms)

loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True)

chrono = exp.chrono()

i = 0

for r in range(args.repeat):
    batch_iterator = iter(loader)

    with chrono.time('train') as t:
        for n in range(args.number):
            batch = next(batch_iterator)
            batch = [item.to(device) for item in batch]

            if args.cuda:
                torch.cuda.synchronize()

    exp.show_eta(r, t)

exp.report()

