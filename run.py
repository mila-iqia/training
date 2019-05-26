import os
import glob
import subprocess
import sys
import time
import json
import copy
import shlex
import traceback
import multiprocessing
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry', action='store_true', default=False)
parser.add_argument('--jobs', type=str, default='baselines.json', help='jobs definition file')
parser.add_argument('--name', type=str, default=None, help='name of the job to run')
parser.add_argument('--show', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--exclude', type=str, default='', help='name of the experience to exclude')

cpu_count = multiprocessing.cpu_count()
device_count = torch.cuda.device_count()
path = os.environ.get('OUTPUT_DIRECTORY', '/tmp')

opt = parser.parse_args()
experiment = open(f'{path}/summary.txt', 'a')

excluded = set(opt.exclude.split(','))

env = os.environ
env['JOB_RUNNER'] = '1'
env['JOB_FILE'] = opt.jobs.split('.')[0]

cgroups = {
    'student': 'cpuset,memory:student',
    'all': 'memory:all'
}


class JobRunnerException(Exception):
    def __init__(self, exceptions, total_processes):
        self.exceptions = exceptions
        self.total_processes = total_processes


def show():
    jobs = json.load(open(opt.jobs, 'r'))
    job_names = [job['name'] for job in jobs]
    print(', '.join(job_names))


def make_configs(args, current=''):
    if len(args) == 0:
        return [current]

    name, values = args.pop()

    if isinstance(values, list):
        configs = []

        for val in values:
            config = make_configs(copy.deepcopy(args), current=f'{current} {name} {val}')
            configs.extend(config)

        return configs

    return make_configs(copy.deepcopy(args), current=f'{current} {name} {values}')


def run_job(cmd, config, group, name):
    """ Run a model on each GPUs """
    env['CGROUP'] = group
    env['BENCH_NAME'] = name

    if device_count <= 1 or group == cgroups['all']:
        env['JOB_ID'] = '0'
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(device_count)])
        subprocess.check_call(f"{cmd} {config} --seed {device_count}", shell=True, env=env)
        return

    cmd = f"{cmd} {config}"
    if opt.verbose:
        print(cmd)

    processes = []
    # use all those GPU
    for i in range(device_count):
        env['CGROUP'] = f'{group}{i}'

        cmd = f"{cmd} --seed {i}"
        processes.append(subprocess.Popen(f'JOB_ID={i} CUDA_VISIBLE_DEVICES={i} {cmd}', env=env, shell=True))

    exceptions = []
    for process in processes:
        try:
            return_code = process.wait()

            if return_code != 0:
                exceptions.append(return_code)

        except Exception as e:
            process.kill()
            exceptions.append(e)

    if len(exceptions) > 0:
        raise JobRunnerException(exceptions, device_count)
    return


def run_job_def(definition, name=None):
    if name is not None and definition['name'] != name:
        return

    if definition['name'] in excluded:
        return

    cmd = definition['cmd']
    args = list(definition['args'].items())
    configs = make_configs(args)

    for config in configs:
        if opt.dry:
            print(f"{cmd} {config}")
            continue

        config = config.strip()
        s = time.time()
        try:
            group = cgroups[definition.get('cgroup', 'all')]

            run_job(cmd, config, group, definition['name'])

            msg = f'{cmd} {(time.time() - s) / 60:8.2f} min passed\n'

        except JobRunnerException as e:
            print(' ' * 4 * 3, e)
            print(' ' * 4 * 3, cmd)
            failed = len(e.exceptions)
            total = e.total_processes

            if failed == total:
                msg = f'{cmd} {(time.time() - s) / 60:8.2f} s all failed {total}\n'
            else:
                msg = f'{cmd} {(time.time() - s) / 60:8.2f} s partial failed {failed}/{total}\n'

        except Exception as e:
            traceback.print_exc()
            print(' ' * 4 * 3, cmd)
            msg = f'{cmd} {(time.time() - s) / 60:8.2f} s failed\n'

        print(msg)
        experiment.write(msg)
        experiment.flush()


def run_job_file(name):
    jobs = json.load(open(opt.jobs, 'r'))
    start_all = time.time()

    for job in jobs:
        run_job_def(job, name)

    msg = f'Total Time {time.time() - start_all:8.2f} s\n'
    print(msg)
    experiment.write(msg)
    experiment.close()


if __name__ == '__main__':

    if opt.show:
        show()
    else:
        run_job_file(opt.name)

