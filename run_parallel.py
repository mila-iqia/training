import os
import subprocess
import time
import json
import copy
from typing import List
import traceback
import multiprocessing
import torch
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dry', action='store_true', default=False)
parser.add_argument('--jobs', type=str, default='fast.json', help='jobs definition file')
parser.add_argument('--name', type=str, default=None, help='name of the job to run')
parser.add_argument('--show', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--exclude', type=str, default='', help='name of the experience to exclude')

parser.add_argument('--no-cgexec', action='store_true', help='do not execute inside a cgroup')
parser.add_argument('--no-nocache', action='store_true', help='do not use nocache')

parser.add_argument('--uid', type=int, default=0)
parser.add_argument('--singularity', type=str, default=None, help='singularity image to use')
parser.add_argument('--raise-error', action='store_true', default=False)

# This is not the way you want to do reproducible benchmarks
parser.add_argument('--free-for-all', action='store_true', default=False,
                    help='run all the benchmarks in parallel making all benchs fight for their resources')


cpu_count = multiprocessing.cpu_count()
device_count = torch.cuda.device_count()
path = os.environ.get('OUTPUT_DIRECTORY', '/tmp')

opt = parser.parse_args()
experiment = open(f'{path}/summary.txt', 'a')

excluded = set(opt.exclude.split(','))

env = dict(os.environ)
env['JOB_RUNNER'] = '1'
env['JOB_FILE'] = opt.jobs.split('.')[0]

cgroups = {
    'student': 'cpuset,memory:student',
    'all': 'memory:all'
}

# Lookup configuration file
# -------------------------
job_file = opt.jobs

# check if the file exists in the CWD
if not os.path.exists(job_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    job_file = f'{current_dir}/profiles/{job_file}'
    print(f'using {job_file}')

if not os.path.exists(job_file):
    print(f'{job_file} was not found')
    sys.exit(-1)
# ---

# Setup command prefix
# --------------------
exec_prefix = []

if not opt.no_cgexec:
    exec_prefix.append('cgexec -g $CGROUP')

if not opt.no_nocache:
    exec_prefix.append('nocache')

# I do not really know how singularity & nocache will interact
if opt.singularity is not None:
    exec_prefix.append(f'singularity exec {opt.singularity}')

exec_prefix = ' '.join(exec_prefix)
# ---


class JobRunnerException(Exception):
    def __init__(self, exceptions, total_processes):
        self.exceptions = exceptions
        self.total_processes = total_processes


def show():
    jobs = json.load(open(opt.jobs, 'r'))
    job_names = [job['name'] for job in jobs]
    print(', '.join(job_names))


def generate_argument_lists(args, current=''):
    """Generate the list of arguments to pass to the run script"""
    if len(args) == 0:
        return [current]

    name, values = args.pop()
    if isinstance(values, list):
        configs = []

        for val in values:
            config = generate_argument_lists(copy.deepcopy(args), current=f'{current} {name} {val}')
            configs.extend(config)

        return configs

    return generate_argument_lists(copy.deepcopy(args), current=f'{current} {name} {values}')


def generate_execution_environment(cmd, arguments, group, name):
    """ Run a model on each GPUs """
    job_env = copy.deepcopy(env)
    job_env['BENCH_NAME'] = name

    if group == cgroups['all']:
        job_env['JOB_ID'] = '0'
        job_env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(device_count)])

        prefix = exec_prefix.replace('$CGROUP', group)
        return 'all', [(f"{prefix} {cmd} {arguments} --seed {opt.uid}", job_env)]

    base_cmd = f"{cmd} {arguments}"

    # use all those GPU
    # for i in range(device_count):
    job_env['JOB_ID'] = str(0)
    job_env['CUDA_VISIBLE_DEVICES'] = str(0)
    exec_cmd = f"{exec_prefix} {base_cmd} --seed $SEED"

    return 'student', [(exec_cmd, job_env)]


def generate_jobs(definition, name=None):
    if name is not None and definition['name'] != name:
        return

    if definition['name'] in excluded:
        return

    cmd = definition['cmd']
    args = list(definition['args'].items())
    argument_lists = generate_argument_lists(args)
    jobs = {
        'all': [],
        'student': []
    }

    for argument_list in argument_lists:
        argument_list = argument_list.strip()

        group = cgroups[definition.get('cgroup', 'all')]
        k, new_jobs = generate_execution_environment(cmd, argument_list, group, definition['name'])

        jobs[k].extend(new_jobs)

    return jobs


def run_cmd(cmd, job_env):
    s = time.time()
    try:
        print(f'Running {cmd}')
        subprocess.check_call(cmd, shell=True, env=job_env)
        print(f'{cmd} {(time.time() - s) / 60:8.2f} s SUCCESS\n')

    except subprocess.CalledProcessError as e:
        print(f'{cmd} {(time.time() - s) / 60:8.2f} s FAILED\n')

        print('=' * 80)
        print(e.cmd)
        print('-' * 80)
        print(e.stdout)
        print('-' * 80)
        print(e.stderr)
        print('=' * 80)


def worker_loop(uid, remote, parent_remote):
    parent_remote.close()

    while True:
        tag, msg = remote.recv()

        if tag == 'job':
            cmd, job_env = msg

            # one worker = one GPU
            job_env['JOB_ID'] = str(uid)
            job_env['CUDA_VISIBLE_DEVICES'] = str(uid)
            cmd = cmd.replace('$CGROUP', f'cpuset,memory:student{uid}')
            cmd = cmd.replace('$SEED', str((uid + 1) * 10 + opt.uid))

            run_cmd(cmd, job_env)
        else:
            remote.close()
            return


def run_job_file(name):
    jobs = json.load(open(job_file, 'r'))
    start_all = time.time()

    all_jobs = {
        'all': [],
        'student': []
    }

    for job in jobs:
        jobs = generate_jobs(job, name)
        for k, v in jobs.items():
            all_jobs[k].extend(v)

    remotes, work_remotes = zip(*[multiprocessing.Pipe() for _ in range(device_count)])

    workers = []
    for i, (work_remote, remote) in enumerate(zip(work_remotes, remotes)):
        workers.append(multiprocessing.Process(target=worker_loop, args=(i, work_remote, remote)))

    for worker in workers:
        worker.start()

    for remote in work_remotes:
        remote.close()

    # Jobs that only use a single GPU
    for cmd, job_env in all_jobs['student']:
        for remote in remotes:
            remote.send(('job', (cmd, job_env)))

    for remote in remotes:
        remote.send(('stop', None))

    # wait for jobs to finish
    for worker in workers:
        worker.join()

    # Jobs that use all the GPUs
    for cmd, job_env in all_jobs['all']:
        run_cmd(cmd, job_env)

    print(f'Total Time {time.time() - start_all:8.2f} s\n')


if __name__ == '__main__':

    if opt.show:
        show()
    else:
        run_job_file(opt.name)

