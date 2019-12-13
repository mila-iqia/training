from argparse import ArgumentParser, REMAINDER
import json
import os
import sys
import subprocess
from milabench.perf import *
import numpy as np


def argunents():
    parser = parser_base('Scaling Task')
    return parser


def launch_distributed(script, args, other_args):
    args.devices = args.devices.split(',')

    processes = []
    job_env = os.environ.copy()
    for rank, device_id in enumerate(args.devices):
        cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, "-u"]
        cmd.append(script)
        cmd.append('--distributed_dataparallel')
        cmd.extend(('--rank', str(rank)))
        cmd.extend(('--world-size', str(len(args.devices))))
        cmd.extend(('--dist-backend', 'nccl'))
        cmd.extend(('--dist-url', 'tcp://localhost:8181'))
        cmd.extend(('--batch-size', str(args.batch_size)))
        # Might be useful to throw IO into the mix with data loading
        # cmd.extend(('--workers', str(args.workers)))
        # cmd.extend(('--seed', str(args.seed)))
        cmd.extend(('--number', str(args.number)))
        cmd.extend(other_args)

        process = subprocess.Popen(' '.join(cmd), env=job_env, shell=True)
        processes.append(process)

    errors = []

    for process in processes:
        process.wait()

        if process.returncode != 0:
            errors.append((process.returncode, cmd))

    for error in errors:
        print(error)

    return len(errors)


def main():
    exp = Experiment(__file__)
    args, other_args = exp.get_arguments(argunents(), show=True, allow_unknown=True)
    assert args.cuda, 'Need GPUs for this Benchmark'

    device_count = int(args.devices) or torch.cuda.device_count()
    args.repeat = device_count
    _ = exp.get_device()
    chrono = exp.chrono()

    scaling = []
    tmp = os.environ.get('TEMP_DIRECTORY', '/tmp')
    script = f'{os.path.dirname(__file__)}/micro_bench.py'

    for i in range(device_count):
        with chrono.time('train') as t:
            args.devices = ','.join([str(d) for d in range(i + 1)])

            rc = launch_distributed(script, args, other_args)

            if rc == 0:
                report = json.load(open(f'{tmp}/overall_report.json', 'r'))
                world_size = report['world_size']
                speed = report['speed']

                assert i + 1 == world_size
                scaling.append((world_size, speed))
            else:
                scaling.append((i + 1, False))

        exp.show_eta(i, t)

    # sort by world size
    # should already be sorted
    scaling.sort(key=lambda x: x[0])

    world_size1, speed1 = scaling[0]
    assert world_size1 == 1

    for world_size, speed in scaling:
        vcd = ','.join(map(str, range(world_size)))
        if speed is False:
            exp.report(results={
                'completed': False,
                'vcd': vcd,
                'train_item': {},
            })
        else:
            speed_up = speed / speed1
            efficiency = speed_up / world_size
            exp.report(results={
                'completed': True,
                'train_item': {
                    'avg': efficiency,
                    'max': efficiency,
                    'min': efficiency,
                    'range': 0,
                    'sd': 0,
                    'unit': '%',
                },
                'vcd': vcd,
            })

    if any(speed is False for world_size, speed in scaling):
        sys.exit(1)


if __name__ == '__main__':
    main()
