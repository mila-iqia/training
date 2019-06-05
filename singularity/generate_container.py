singularity_base = """
Bootstrap: docker
From: {base_img}

%post
    apt-get -y update
    apt-get -y install {apt_packages}

    pip install --user --upgrade setuptools

    pip install -e git+git://github.com/mila-iqia/training.git#egg=master\\&subdirectory=common
    pip install Cython
{pip_packages}

    {more}
"""

docker_base = """
FROM {base_img}

RUN apt-get -y update
RUN apt-get -y install {apt_packages}

RUN pip install --upgrade setuptools

RUN pip install -e git+git://github.com/mila-iqia/training.git#egg=master\\&subdirectory=common
RUN pip install Cython
{pip_packages}

{more}
"""

generate_docker = False
generate_singularity = True


# we want to know exactly which packaged failed to install
def make_pip_install(name, docker=False):
    pip_packages = open(name, 'r').read().split('\n')
    prefix = '   '
    if docker:
        prefix = 'RUN '

    pip_packages = map(lambda p: p.strip(), pip_packages)
    pip_packages = filter(lambda p: len(p) > 0, pip_packages)
    return '\n'.join([f'{prefix} pip install --no-deps {p}' for p in pip_packages])


def make_file(name, script, base_img, apt_packages, pip_packages, more=''):
    script = (script
        .replace('{base_img}', base_img)
        .replace('{apt_packages}', apt_packages)
        .replace('{pip_packages}', pip_packages)
        .replace('{more}', more))

    with open(name, 'w') as f:
        f.write(script)

    return script


apt_packages = (open('../apt_packages', 'r')
    .read()
    .replace('\n', ' '))


# ROCm Image
# ---------------------------------------------------
# Remove all the reference to useless python versions
# and make python3.6 the default
more_rocm = ' && '.join([
    'rm /usr/bin/python3',
    'rm /usr/bin/python',
    'ln -s /usr/bin/python3.6 /usr/bin/python',
    'ln -s /usr/bin/python3.6 /usr/bin/python3'
])

if generate_singularity:
    make_file(
        'Singularity.rocm',
        singularity_base,
        'rocm/pytorch:rocm2.2_ubuntu16.04_py3.6_pytorch',
        apt_packages,
        make_pip_install('../requirements.txt'),
        more=more_rocm
    )

if generate_docker:
    make_file(
        'Dockerfile.rocm',
        docker_base,
        'rocm/pytorch:rocm2.2_ubuntu16.04_py3.6_pytorch',
        apt_packages,
        make_pip_install('../requirements.txt'),
        more=f'RUN {more_rocm}'
    )

# Power9
# ---------------------------------------------------

if generate_singularity:
    make_file(
        'Singularity.power9',
        singularity_base,
        'ibmcom/powerai:1.5.4-all-ubuntu18.04-py3',
        apt_packages,
        make_pip_install('../requirements_ppc.txt')
    )

if generate_docker:
    make_file(
        'Dockerfile.power9',
        docker_base,
        'ibmcom/powerai:1.5.4-all-ubuntu18.04-py3',
        apt_packages,
        make_pip_install('../requirements_ppc.txt'),
    )

# NVIDIA
# ---------------------------------------------------
if generate_singularity:
    make_file(
        'Singularity.nvidia',
        singularity_base,
        'pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime',
        apt_packages,
        make_pip_install('../requirements.txt')
    )

if generate_docker:
    make_file(
        'Dockerfile.nvidia',
        docker_base,
        'pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime',
        apt_packages,
        make_pip_install('../requirements.txt')
    )

