class CMLExperimentMock:
    def __init__(*args, **kwargs):
        pass

    def log_metric(self, a, b):
        pass

    def log_parameters(self, a):
        pass


try:
    from comet_ml import Experiment as CMLExperiment
except Exception as e:
    print(e)
    CMLExperiment = CMLExperimentMock


import argparse
from argparse import Namespace

import benchutils.arguments as bench_args
from benchutils.versioning import get_file_version
from benchutils.chrono import MultiStageChrono

import hashlib
import socket
import os
import json
import torch
import array
from typing import *


def get_arguments(parser, *args, **kwargs):

    args = bench_args.get_arguments(parser, *args, **kwargs)
    args.jr_id = os.environ.get('JOB_ID', 0)
    args.vcd = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
    args.cpu_cores = int(os.environ.get('CPU_COUNT', 32))

    return args


MultiStageChrono = MultiStageChrono

excluded_arguments = {
    'report',
    'seed'
}

not_parameter = {
    'gpu',
    'batch_size',
    'num-processes',
    'workers',
    'cuda',
    'report',
    'ngpu',
    'data',
    'no_cuda',
    'dataset_dir'
}


def get_experience_descriptor(name: str) -> Tuple[str, str]:
    version = get_file_version(name)[:10]
    name = '_'.join(name.split('/')[-4:])
    return name, version


class RingBuffer:
    types = {
        torch.float16: 'f',  # 4
        torch.float32: 'f',  # 4
        torch.float64: 'd',  # 8

        torch.int8:  'b',    # 1
        torch.int16: 'h',    # 2
        torch.int32: 'l',    # 4
        torch.int64: 'q',    # 8

        torch.uint8:  'B',   # 1
        # torch.uint16: 'H',   # 2
        # torch.uint32: 'L',   # 4
        # torch.uint64: 'Q',   # 8
    }

    def __init__(self, size, dtype, default_val=0):
        self.array = array.array(self.types[dtype], [default_val] * size)
        self.capacity = size
        self.offset = 0

    def __getitem__(self, item):
        return self.array[item % self.capacity]

    def __setitem__(self, item, value):
        self.array[item % self.capacity] = value

    def append(self, item):
        self.array[self.offset % self.capacity] = item
        self.offset += 1

    def to_list(self):
        if self.offset < self.capacity:
            return list(self.array[:self.offset])
        else:
            end_idx = self.offset % self.capacity
            return list(self.array[end_idx: self.capacity]) + list(self.array[0:end_idx])

    def __len__(self):
        return min(self.capacity, self.offset)

    def last(self):
        if self.offset == 0:
            return None

        return self.array[(self.offset - 1) % self.capacity]


class Experiment:
    """ Store all the information we care about during an experiment
            chrono      : Performance timer
            args        : argument passed to the script
            name        : name of the experiment
            batch_loss_buffer : Batch loss values (just making sure we do not get NaNs
            epoch_loss_buffer:  Epoch loss values same as batch loss
    """

    def __init__(self, module, skip_obs=10):
        self.name, self.version = get_experience_descriptor(module)
        self._chrono = None
        self.skip_obs = skip_obs
        self.args = None
        self.batch_loss_buffer = RingBuffer(100, torch.float32)
        self.epoch_loss_buffer = RingBuffer(10, torch.float32)
        self.metrics = {}

        self.remote_logger = None

        try:
            self.remote_logger = CMLExperiment(
                api_key=os.environ.get("CML_API_KEY"),
                project_name=self.name,
                workspace=os.environ.get("CML_WORKSPACE")
            )
        except Exception as e:
            print(e)
            self.remote_logger = CMLExperimentMock()

    def get_arguments(self, parser, *args, **kwargs):
        self.args = get_arguments(parser, *args, **kwargs)
        self._chrono = MultiStageChrono(name=self.name, skip_obs=self.skip_obs, sync=get_sync(self.args))

        return self.args

    def chrono(self):
        return self._chrono

    def log_batch_loss(self, val):
        self.batch_loss_buffer.append(val)
        self.remote_logger.log_metric('batch_loss', val)

    def log_epoch_loss(self, val):
        self.epoch_loss_buffer.append(val)
        self.remote_logger.log_metric('epoch_loss', val)

    def log_metric(self, name, val):
        metric = self.metrics.get(name)
        if metric is None:
            metric = RingBuffer(10, torch.float32)
            self.metrics[name] = metric
        metric.append(val)
        self.remote_logger.log_metric(name, val)

    def report(self):
        make_report(
            self._chrono,
            self.args,
            self.version,
            self.batch_loss_buffer,
            self.epoch_loss_buffer,
            {name: val.to_list() for name, val in self.metrics.items()},
            self.remote_logger
        )

    def get_device(self):
        return init_torch(self.args)

    def get_time(self, stream):
        if stream.avg == 0:
            return stream.val
        return stream.avg

    def show_eta(self, epoch_id, timer, msg=''):
        if msg:
            msg = ' | ' + msg

        loss = self.batch_loss_buffer.last()
        if loss is not None:
            loss = f'| Batch Loss {loss:8.4f}'
        else:
            loss = ''

        print(f'[{epoch_id:3d}/{self.args.repeat:3d}] '
              f'| ETA: {self.get_time(timer) * (self.args.repeat - (epoch_id + 1)) / 60:6.2f} min ' + loss + msg)


def parser_base(description=None, **kwargs):
    import os
    parser = argparse.ArgumentParser(description, **kwargs)

    nproc = os.environ.get('PROC_COUNT')
    if nproc is None:
        nproc = 0

    parser.add_argument('--batch-size', '-b', type=int, help='batch size', default=1)
    parser.add_argument('--cuda', action='store_true', dest='cuda', help='enable cuda', default=True)
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable cuda')
    parser.add_argument('--workers', '-j', type=int, help='number of workers/processors to use', default=nproc)
    parser.add_argument('--seed', '-s', type=int, help='seed to use', default=0)
    parser.add_argument('--devices', type=int, help='number of device available', default=1)

    parser.add_argument('--jr_id', type=int, default=0)
    parser.add_argument('--vcd', type=int, default=0)
    parser.add_argument('--cpu-cores', type=int, default=0)

    return parser


def make_report(chrono: MultiStageChrono, args: Namespace, version: str, batch_loss: RingBuffer, epoch_loss: RingBuffer, metrics, remote_logger):
    if args is not None:
        args = args.__dict__
    else:
        args = {}

    if args['report'] is None:
        args['report'] = os.environ.get('REPORT_PATH')

    filename = args['report']
    # Each GPU has its report we will consolidate later
    filename = f'{filename}_{args["jr_id"]}.json'

    args['version'] = version

    unique_id = hashlib.sha256()

    # make it deterministic
    items = list(args.items())
    items.sort()

    for k, w in items:
        if k in not_parameter:
            continue

        unique_id.update(str(k).encode('utf-8'))
        unique_id.update(str(w).encode('utf-8'))

    # we do not want people do modify our shit if the id do not match then they get disqualified
    args['unique_id'] = unique_id.hexdigest()

    # Try to identify vendors so we can find them more easily
    if args['cuda']:
        args['gpu'] = get_gpu_name()

    args['hostname'] = socket.gethostname()
    args['batch_loss'] = batch_loss.to_list()
    args['epoch_loss'] = epoch_loss.to_list()
    args['metrics'] = metrics

    for excluded in excluded_arguments:
        args.pop(excluded, None)

    remote_logger.log_parameters(args)
    report_dict = chrono.to_dict(args)

    # train is the default name for batched stuff
    if 'train' in report_dict:
        train_data = report_dict['train']

        item_count = report_dict['batch_size'] * report_dict['number']
        min_item = item_count / train_data['max']
        max_item = item_count / train_data['min']

        train_item = {
            'avg': item_count / train_data['avg'],
            'max': max_item,
            'min': min_item,
            'range': max_item - min_item,
            'unit': 'items/sec'
        }

        report_dict['train_item'] = train_item

    print('-' * 80)
    json_report = json.dumps(report_dict, sort_keys=True, indent=4, separators=(',', ': '))
    print(json_report)

    if not os.path.exists(filename):
        report_file = open(filename, 'w')
        report_file.write('[')
        report_file.close()

    report_file = open(filename, 'a')
    report_file.write(json_report)
    report_file.write(',')
    report_file.close()

    print('-' * 80)


def get_gpu_name():
    import torch
    current_device = torch.cuda.current_device()
    return torch.cuda.get_device_name(current_device)


def get_sync(args):
    import torch

    if args.cuda:
        return lambda: torch.cuda.synchronize()

    return lambda: None


def init_torch(args):
    import torch
    import numpy as np

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_cores)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    return torch.device("cuda" if args.cuda else "cpu")


def torch_train(model, optimizer, criterion, dataloader, chrono, args):
    loss = float('+inf')
    model.train()

    for i in range(args.repeat):

        with chrono.time('task'):
            for batch_id, data in enumerate(dataloader):
                if batch_id >= args.number:
                    break

                x, y = data

                optimizer.zero_grad()

                # Forward pass
                output = criterion(model(x), y)
                loss = output.item()

                # Backward pass
                output.backward()

                # Apply gradients
                output.step()

    return loss

