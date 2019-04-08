import os
from perf import *
import time


class BenchTrainer:

    def __init__(self, script_path, model, criterion, optimizer):
        self.name, self.version = get_experience_descriptor(script_path)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.chrono = None
        self.report = os.environ.get('REPORT_PATH')

    def benchmark(self, args, dataloader, repeat: int = None, number: int = None):
        print('Starting Benchmark')
        device = init_torch(args)
        self.chrono = MultiStageChrono(name=self.name, sync=get_sync(args))
        start_time = time.time()

        if repeat is None:
            repeat = args.repeat

        if number is None:
            number = args.number

        for i in range(repeat):
            losses = 0

            with self.chrono.time('training') as step_time:
                for batch_id, (x, y) in enumerate(dataloader):
                    if batch_id > number:
                        break

                    x = x.to(device)

                    self.model.train()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                    losses += loss.data.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            elapsed = step_time.avg
            if elapsed == 0:
                elapsed = step_time.val

            print(f'[{i + 1:3d}/{repeat}] '
                  f'Elapsed: {(time.time() - start_time) / 60:6.2f} min ' + ' ' * 4 + \
                  f'ETA: {elapsed * (repeat - i - 1) / 60:6.2f} min')

        make_report(self.chrono, args, self.version)


if __name__ == '__main__':
    from torchvision.models import resnet18
    import torch.nn as nn
    import torch.optim
    from torchvision.datasets import FakeData
    import torchvision.transforms as transforms

    parser = parser_base()
    args = get_arguments(parser, show=True)
    args.batch_size = 2
    args.repeat = 10
    args.number = 10

    model = resnet18()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        0.001
    )

    dataset = FakeData(
        size=args.batch_size * args.number,
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.workers,
        shuffle=False,
    )

    bench = BenchTrainer(
        __file__,
        model,
        criterion,
        optimizer
    )

    bench.benchmark(
        args,
        dataloader,
        args.repeat,
        args.number
    )

    print('done')
