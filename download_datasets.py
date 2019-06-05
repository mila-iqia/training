import os
import subprocess
import sys
import time
import argparse

path = os.environ.get('OUTPUT_DIRECTORY', '/tmp')
parser = argparse.ArgumentParser()
parser.add_argument('--sequential', action='store_true', default=False)
parser.add_argument('--singularity', type=str, default=None, help='singularity image to use')
args, _ = parser.parse_known_args()


def isdir(path):
    return path[0] != '.'and os.path.isdir(path)


def make_title(index, name, depth=0):
    indent = ' ' * depth * 4
    title = '{}) {}'.format(index + 1, name.replace('_', ' ').capitalize())
    print('{}{}\n{}{}'.format(indent, title, indent, '-' * len(title)))


sys.stderr = sys.stdout
base_path = '/'.join(__file__.split('/')[:-1])
experiment = open(f'{path}/download_summary.txt', 'a')

start_all = time.time()

# if singularity is used to run we also need to download with singularity
exec_prefix = ()
if args.singularity is not None:
    exec_prefix = ('singularity', 'exec', args.singularity)

# Task
scripts = []


def run_script(download_script):
    s = time.time()
    try:
        arguments = ' '.join(exec_prefix + (download_script,))
        subprocess.check_call(arguments, shell=True, env=os.environ)
        experiment.write(f'{download_script} {time.time() - s} passed\n')

    except Exception as e:
        print(' ' * 4 * 3, e)
        print(' ' * 4 * 3, download_script)
        experiment.write(f'{download_script} {time.time() - s} failed\n')


# Task
for i1, task in enumerate(filter(isdir, os.listdir(base_path))):
    models_path = os.path.join(base_path, task)
    make_title(i1, task)

    i2 = 0
    for model in os.listdir(models_path):
        frameworks_path = os.path.join(models_path, model)

        if os.path.isdir(frameworks_path):
            for script_name in os.listdir(frameworks_path):
                download_script = os.path.join(frameworks_path, script_name)

                if download_script.endswith('download_dataset.sh'):
                    if args.sequential:
                        run_script(download_script)
                    else:
                        scripts.append(download_script)


if not args.sequential:
    from multiprocessing import Pool, cpu_count

    pool = Pool(cpu_count())
    pool.map(run_script, scripts)


data = f'Total Time {time.time() - start_all}'
print(data)
experiment.write(data)
experiment.close()
