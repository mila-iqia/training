import os
import subprocess
import time
import json
import copy
import traceback
import multiprocessing
import torch
import argparse

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

env = os.environ
env['JOB_RUNNER'] = '1'
env['JOB_FILE'] = opt.jobs.split('.')[0]

cgroups = {
    'student': 'cpuset,memory:student',
    'all': 'memory:all'
}


exec_prefix = []

if not opt.no_cgexec:
    exec_prefix.append('cgexec -g $CGROUP')

if not opt.no_nocache:
    exec_prefix.append('nocache')

# I do not really know how singularity & nocache will interact
if opt.singularity is not None:
    exec_prefix.append(f'singularity exec {opt.singularity}')

exec_prefix = ' '.join(exec_prefix)


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
    env['BENCH_NAME'] = name
    env['RUN_ID'] = str(opt.uid)

    if group == cgroups['all']:
        env['JOB_ID'] = '0'
        env['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(device_count)])
        prefix = exec_prefix.replace('$CGROUP', group)

        subprocess.check_call(f"{prefix} {cmd} {config} --seed {opt.uid + device_count}", shell=True, env=env)
        return

    cmd = f"{cmd} {config}"
    if opt.verbose:
        print(cmd)

    processes = []
    # use all those GPU
    for i in range(device_count):
        prefix = exec_prefix.replace('$CGROUP', f'{group}{i}')

        exec_cmd = f"{prefix} {cmd} --seed {opt.uid + (i + 1) * 100}"
        processes.append(subprocess.Popen(f'JOB_ID={i} CUDA_VISIBLE_DEVICES={i} {exec_cmd}', env=env, shell=True))

    exceptions = []
    for process in processes:
        try:
            return_code = process.wait()

            if return_code != 0:
                exceptions.append(return_code)

        except Exception as e:
            process.kill()
            exceptions.append(e)

            if opt.raise_error:
                raise e

    if len(exceptions) > 0:
        raise JobRunnerException(exceptions, device_count)
    return


def run_job_def(jid, definition, name=None, size=19):
    if name is not None and definition['name'] != name:
        return

    if definition['name'] in excluded:
        return

    progress = f'[{jid:2d}/{size:2d}]'
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

            msg = f'{progress}[  /  ] PASSED | {(time.time() - s) / 60:8.2f} MIN | {cmd}\n'

        except JobRunnerException as e:
            print(' ' * 4 * 3, e)
            print(' ' * 4 * 3, cmd)
            failed = len(e.exceptions)
            total = e.total_processes

            msg = f'{progress}[{failed:2d}/{total:2d}] FAILED | {(time.time() - s) / 60:8.2f} MIN | {cmd}\n'

            if opt.raise_error:
                raise e

        except Exception as e:
            traceback.print_exc()
            print(' ' * 4 * 3, cmd)
            msg = f'{progress}[  /  ] FAILED | {(time.time() - s) / 60:8.2f} MIN | {cmd}\n'

            if opt.raise_error:
                raise e

        print(msg)
        experiment.write(msg)
        experiment.flush()


def run_job_file(name):
    job_file = opt.jobs

    # check if the file exists in the CWD
    if not os.path.exists(job_file):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        job_file = f'{current_dir}/profiles/{job_file}'

    print(f'Using {job_file}')
    jobs = json.load(open(job_file, 'r'))
    start_all = time.time()

    for id, job in enumerate(jobs):
        run_job_def(id + 1, job, name, len(jobs))

    msg = f'Total Time {time.time() - start_all:8.2f} s\n'
    print(msg)
    experiment.write(msg)
    experiment.close()


if __name__ == '__main__':

    if opt.show:
        show()
    else:
        run_job_file(opt.name)

