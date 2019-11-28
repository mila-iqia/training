from argparse import REMAINDER
import json
import os
import sys
import subprocess

import numpy as np

from perf import *


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

    device_count = int(args.devices)
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

            assert rc == 0, 'Failed to run distributed script'

            report = json.load(open(f'{tmp}/overall_report.json', 'r'))
            world_size = report['world_size']
            speed = report['speed']

            assert i + 1 == world_size
            scaling.append((world_size, speed))

        exp.show_eta(i, t)

    # sort by world size
    # should already be sorted
    scaling.sort(key=lambda x: x[0])

    world_size1, speed1 = scaling[0]
    assert world_size1 == 1

    if len(scaling) == 1:
        print('No Multi GPU cannot compute GPU scaling')
        print(json.dumps(report, indent=2))
    else:
        all_efficiency = []

        for world_size, speed in scaling[1:]:
            speed_up = speed / speed1

            # with 8 GPUs the speed up should be close 8x
            efficiency = world_size / speed_up
            all_efficiency.append(efficiency)

        results = np.array(all_efficiency)

        train_item = {
            'avg': results.mean(),
            'max': results.max(),
            'min': results.min(),
            'range': results.max() - results.min(),
            'sd': results.std(),
            'unit': '%'
        }

        exp.report(override=train_item)


if __name__ == '__main__':
    main()
