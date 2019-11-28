import torch
import torchvision
import random
import time
import argparse
import os
import sys
import math
import torch.nn as nn
from fp16util import network_to_half, get_param_copy
import json


tmp = os.environ.get('TEMP_DIRECTORY', '/tmp')


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def get_network(net):
    classification_models = torchvision.models.__dict__
    segmentation_models = torchvision.models.segmentation.__dict__

    if net in classification_models:
        return classification_models[net]().cuda()

    if net in segmentation_models:
        return segmentation_models[net]().cuda()

    print("ERROR: not a supported model.")
    sys.exit(1)


def forwardbackward(inp, optimizer, network, target):
    optimizer.zero_grad()
    out = network(inp)
    # WIP: googlenet, deeplabv3_*, fcn_* missing log_softmax for this to work
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    optimizer.step()


def rendezvous(distributed_parameters):
    print("Initializing process group...")
    torch.distributed.init_process_group(
        backend=distributed_parameters['dist_backend'],
        init_method=distributed_parameters['dist_url'],
        rank=distributed_parameters['rank'],
        world_size=distributed_parameters['world_size']
    )
    print("Rendezvous complete. Created process group...")


def run_benchmarking(net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids=None,
                     distributed_parameters=None):
    if device_ids:
        torch.cuda.set_device("cuda:%d" % device_ids[0])
    else:
        torch.cuda.set_device("cuda:0")

    network = get_network(net)
    if run_fp16:
        network = network_to_half(network)

    if dataparallel:
        network = torch.nn.DataParallel(network, device_ids=device_ids)
        num_devices = len(device_ids) if device_ids is not None else torch.cuda.device_count()

    elif distributed_dataparallel:
        rendezvous(distributed_parameters)
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=device_ids)
        num_devices = len(device_ids) if device_ids is not None else torch.cuda.device_count()

    else:
        num_devices = 1

    if net == "inception_v3":
        inp = torch.randn(batch_size, 3, 299, 299, device="cuda")
    else:
        inp = torch.randn(batch_size, 3, 224, 224, device="cuda")

    if run_fp16:
        inp = inp.half()

    target = torch.randint(0, 1, size=(batch_size,), device='cuda')  # torch.arange(batch_size, device="cuda")

    param_copy = network.parameters()
    if run_fp16:
        param_copy = get_param_copy(network)

    optimizer = torch.optim.SGD(param_copy, lr=0.01, momentum=0.9)

    ## warmup.
    print("INFO: running forward and backward for warmup.")
    forwardbackward(inp, optimizer, network, target)
    forwardbackward(inp, optimizer, network, target)

    time.sleep(1)
    torch.cuda.synchronize()

    ## benchmark.
    print("INFO: running the benchmark..")
    tm = time.time()
    for i in range(iterations):
        forwardbackward(inp, optimizer, network, target)

    torch.cuda.synchronize()

    tm2 = time.time()
    time_per_batch = (tm2 - tm) / iterations
    rank = distributed_parameters.get('rank', -1)
    world_size = distributed_parameters.get('world_size', 1)

    process_report = {
        'model': net,
        'rank': rank,
        'num_device': num_devices,
        'batch_size': batch_size,
        'batch_time': time_per_batch,
        'speed': batch_size / time_per_batch
    }

    with open(f'{tmp}/process_report_{rank}.json', 'w') as report:
        json.dump(process_report, report)

    if rank == 0:
        overall_report = {
            'world_size': world_size,
            'batch_size': batch_size * world_size,
            'batch_time': time_per_batch,
            'speed': batch_size * world_size / time_per_batch
        }
        with open(f'{tmp}/overall_report.json', 'w') as report:
            json.dump(overall_report, report)


def main():
    net = args.network
    batch_size = args.batch_size
    iterations = args.number
    run_fp16 = args.fp16
    dataparallel = args.dataparallel
    distributed_dataparallel = args.distributed_dataparallel
    device_ids_str = args.device_ids

    if args.device_ids:
        device_ids = [int(x) for x in device_ids_str.split(",")]

    else:
        device_ids = None

    distributed_parameters = dict()
    distributed_parameters['rank'] = args.rank
    distributed_parameters['world_size'] = args.world_size
    distributed_parameters['dist_backend'] = args.dist_backend
    distributed_parameters['dist_url'] = args.dist_url

    # Some arguments are required for distributed_dataparallel
    if distributed_dataparallel:
        assert args.rank is not None and \
               args.world_size is not None and \
               args.dist_backend is not None and \
               args.dist_url is not None, "rank, world-size, dist-backend and dist-url are required arguments for distributed_dataparallel"

    run_benchmarking(net, batch_size, iterations, run_fp16, dataparallel, distributed_dataparallel, device_ids,
                     distributed_parameters)


if __name__ == '__main__':
    models = [
        'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
        'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'shufflenet',
        'shufflenet_v2_x05', 'shufflenet_v2_x10', 'shufflenet_v2_x15', 'SqueezeNet',
        'SqueezeNet1.1', 'densenet121', 'densenet169', 'densenet201', 'densenet161',
        'inception', 'inception_v3', 'resnext50', 'resnext101', 'mobilenet_v2', 'googlenet',
        'deeplabv3_resnet50', 'deeplabv3_resnet101', 'fcn_resnet50', 'fcn_resnet101'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=models,required=True,
                        help="Network to run.")
    parser.add_argument("--batch-size", type=int, required=False, default=64,
                        help="Batch size (will be split among devices used by this invocation)")
    parser.add_argument("--number", type=int, required=False, default=20,
                        help="Iterations")
    parser.add_argument("--fp16", type=int, required=False, default=0,
                        help="FP16 mixed precision benchmarking")
    parser.add_argument("--dataparallel", action='store_true', required=False,
                        help="Use torch.nn.DataParallel api to run single process on multiple devices. Use only one of --dataparallel or --distributed_dataparallel")
    parser.add_argument("--distributed_dataparallel", action='store_true', required=False,
                        help="Use torch.nn.parallel.DistributedDataParallel api to run on multiple processes/nodes. The multiple processes need to be launched manually, this script will only launch ONE process per invocation. Use only one of --dataparallel or --distributed_dataparallel")
    parser.add_argument("--device_ids", type=str, required=False, default=None,
                        help="Comma-separated list (no spaces) to specify which HIP devices (0-indexed) to run dataparallel or distributedDataParallel api on. Might need to use HIP_VISIBLE_DEVICES to limit visiblity of devices to different processes.")
    parser.add_argument("--rank", type=int, required=False, default=None,
                        help="Rank of this process. Required for --distributed_dataparallel")
    parser.add_argument("--world-size", type=int, required=False, default=None,
                        help="Total number of ranks/processes. Required for --distributed_dataparallel")
    parser.add_argument("--dist-backend", type=str, required=False, default=None,
                        help="Backend used for distributed training. Can be one of 'nccl' or 'gloo'. Required for --distributed_dataparallel")
    parser.add_argument("--dist-url", type=str, required=False, default=None,
                        help="url used for rendezvous of processes in distributed training. Needs to contain IP and open port of master rank0 eg. 'tcp://172.23.2.1:54321'. Required for --distributed_dataparallel")

    args = parser.parse_args()

    main()

