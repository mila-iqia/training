import sys
import subprocess
from argparse import ArgumentParser, REMAINDER
from perf import *
import numpy as np


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('--devices', type=str, default=None)
    parser.add_argument('script', type=str)
    parser.add_argument('args', nargs=REMAINDER)
    return parser.parse_args(args)


def launch_distributed(args):
    args = parse_args(args)
    processes = []

    args.devices = args.devices.split(',')
    print(args)

    job_env = os.environ.copy()
    for rank, device_id in enumerate(args.devices):
        cmd = [f'CUDA_VISIBLE_DEVICES={device_id}', sys.executable, "-u"]
        cmd.append(args.script)
        cmd.append('--distributed_dataparallel')
        cmd.extend(('--rank', str(rank)))
        cmd.extend(('--world-size', str(len(args.devices))))
        cmd.extend(('--dist-backend', 'nccl'))
        cmd.extend(('--dist-url', 'tcp://localhost:8181'))
        cmd.extend(args.args)

        print(cmd)
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


def child_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str,required=True,
                        help="Network to run.")
    parser.add_argument("--fp16", type=int, required=False, default=0,
                        help="FP16 mixed precision benchmarking")
    return parser


def main():
    exp = Experiment(__file__)
    device_count = torch.cuda.device_count()

    parser = child_arguments()
    args = exp.get_arguments(parser, show=True)
    args.repeat = device_count
    _ = exp.get_device()
    chrono = exp.chrono()
    tmp = os.environ.get('TEMP_DIRECTORY', '/tmp')
    args = sys.argv

    script = f'{os.path.dirname(__file__)}/micro_bench.py'

    scaling = []
    args = [
        '--network', args.network,
        '--number', args.number,
        '--batch-size', args.batch_size,
        '--fp16', args.fp16
    ]

    for i in range(device_count):
        with chrono.time('train') as t:
            rc = launch_distributed([
                '--devices', ','.join(range(i)),
                script,
            ].extend(args))

            assert rc == 0, 'Failed to run distributed script'

            report = json.load(open(f'{tmp}/overall_report.json', 'r'))

            world_size = report['world_size']
            speed = report['speed']

            assert i == world_size
            scaling.append((world_size, speed))

        exp.show_eta(i, t)

    # sort by world size
    # should already be sorted
    scaling.sort(key=lambda x: x[0])

    world_size1, speed1 = scaling[0]
    assert world_size1 == 1

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
