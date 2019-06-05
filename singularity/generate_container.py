singularity_base = """
Bootstrap: docker
From: {base_img}

%post
    apt-get -y update
    apt-get -y install {apt_packages}

    pip install Cython
    pip install --no-deps {pip_packages}

    {more}

%environment
    export BASE=/data/
"""

docker_base = """
FROM {base_img}

RUN apt-get -y update
RUN apt-get -y install {apt_packages}

RUN pip install Cython
RUN pip install --no-deps {pip_packages}

{more}

ENV export BASE=/data/
"""

generate_docker = False
generate_singularity = True


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

pip_packages = (open('../requirements.txt', 'r')
    .read()
    .replace('\n', ' '))

pip_packages_p9 = (open('../requirements_ppc.txt', 'r')
    .read()
    .replace('\n', ' '))

# ROCm Image
# ---------------------------------------------------
# Remove all the reference to useless python versions
# and make python3.6 the default
more_rocm = ' && '.join([
    'rm /usr/bin/python3'
    'rm /usr/bin/python'
    'ln -s /usr/bin/python3.6 /usr/bin/python'
    'ln -s /usr/bin/python3.6 /usr/bin/python3'
])

if generate_singularity:
    make_file(
        'rocm_singularity.recipe',
        singularity_base,
        'rocm/pytorch:rocm2.2_ubuntu16.04_py3.6_pytorch',
        apt_packages,
        pip_packages,
        more=more_rocm
    )

if generate_docker:
    make_file(
        'Dockerfile.rocm',
        docker_base,
        'rocm/pytorch:rocm2.2_ubuntu16.04_py3.6_pytorch',
        apt_packages,
        pip_packages,
        more=f'RUN {more_rocm}'
    )

# Power9
# ---------------------------------------------------

if generate_singularity:
    make_file(
        'power9_singularity.recipe',
        singularity_base,
        'ibmcom/powerai:1.5.4-all-ubuntu18.04-py3',
        apt_packages,
        pip_packages_p9
    )

if generate_docker:
    make_file(
        'Dockerfile.power9',
        docker_base,
        'ibmcom/powerai:1.5.4-all-ubuntu18.04-py3',
        apt_packages,
        pip_packages_p9
    )

# NVIDIA
# ---------------------------------------------------
if generate_singularity:
    make_file(
        'nvidia_singularity.recipe',
        singularity_base,
        'pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime',
        apt_packages,
        pip_packages
    )

if generate_docker:
    make_file(
        'Dockerfile.nvidia',
        docker_base,
        'pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime',
        apt_packages,
        pip_packages
    )

