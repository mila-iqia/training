

**IMPORTANT: This repository is deprecated.**

**The benchmarks are now located at: https://github.com/mila-iqia/milabench**


# Training Benchmarks

* Tested on **python 3.6** with **pytorch 1.3**

## Install

* Install the software and dependencies using Anaconda

```bash
$ ./install_dependencies.sh
$ ./install_conda.sh

# reload bash with anaconda
$ exec bash
$ conda activate mlperf
$ ./install_python_dependencies.sh

# Install pytorch
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

* Set up the cgroups

```bash
$ ./cgroup_setup.sh
```

* Download the datasets

```bash
$ export BASE=~/data/
$ ./download_datasets.sh
```

### Installing on AMD

The baselines should work on AMD GPUs, provided one installs a compatible version of PyTorch. For AMD GPUs, instead of `conda install pytorch`, follow these instructions: https://github.com/ROCmSoftwarePlatform/pytorch/wiki/Building-PyTorch-for-ROCm#option-4-install-directly-on-host

## Executing the benchmarks

The benchmarks should be run in the conda environment created during the installation.

```bash
# Set up environment and necessary variables
$ conda activate mlperf    #if not already active
$ export BASE=~/data/
$ export OUTDIR=~/results-$(date '+%Y-%m-%d.%H:%M:%S')/

# To run only once:
$ ./run.sh --jobs baselines.json --outdir $OUTDIR

# To run ten times:
$ ./run_10.sh --jobs baselines.json --outdir $OUTDIR
```

The test results will be stored as json files in the specified outdir, one file for each test. A result will have a name such as `baselines.vae.R0.D0.20200106-160000-123456.json`, which means the test named `vae` from the jobs file `profiles/baselines.json`, run 0, device 0, and then the date and time. If the tests are run 10 times and there are 8 GPUs, you should get 80 of these files for each test (`R0` through `R9` and `D0` through `D7`). If a test fails, the filename will also contain the word `FAIL` (note: run number N corresponds to the option `--uid N` to `run.sh`).

## Reporting the results

The `mlbench-report` tool (the install procedure will install it automatically) can be used to generate an HTML report:

```bash
$ mlbench-report --jobs baselines --reports $OUTDIR --gpu-model RTX --title "Results for RTX" --html report.html
```

You may open the HTML report in any browser. It reports numeric performance results as compared to existing baselines for the chosen GPU model, results for all pass/fail criteria, a global score, and some handy tables comparing all GPUs to each other and highlighting performance discrepancies between them.

The command also accepts a `--price` argument (in dollars) to compute the price/score ratio.

### Running a specific test

To run a specific test, for example the vae test:

```bash 
./run.sh --jobs baselines.json --name vae --outdir $OUTDIR
```

This is useful if one or more of the tests fail.

## Baseline tweaks

You can create tweaked baselines by modifying a copy of `baselines.json`. These tweaked baselines may be used to either test something different, debug, or demonstrate further capacities, if needed.

```bash
$ cp profiles/baselines.json profiles/tweaked.json

# modify tweaked.json to reflect the device capacity

$ ./run.sh --jobs tweaked.json --outdir $OUTDIR  # run the tweaked version
```

# Docker [Experimental]

You can use cgroups and docker using the script below.

```bash
$ docker build -t my_docker -f Dockerfile .

$ sudo docker run --cap-add=SYS_ADMIN --security-opt=apparmor:unconfined -it my_docker

$ conda activate mlperf

$ mount -t tmpfs cgroup_root /sys/fs/cgroup

$ mkdir /sys/fs/cgroup/cpuset
$ mount -t cgroup cpuset -o cpuset /sys/fs/cgroup/cpuset

$ mkdir /sys/fs/cgroup/memory
$ mount -t cgroup memory -o memory /sys/fs/cgroup/memory
```

After this, follow from 'Set up the cgroups' step in the **Install** section above

# Tips

* The benchmark starts with two toy examples to make sure everything is setup properly.

* Each bench run `N_GPU` times in parallel with only `N_CPU / N_GPU` and `RAM / N_GPU` to simulate multiple users.
  * if your machine has 16 GPUs and 32 cores, the bench will run in parallel 16 times with only 2 cores for each GPUs.

* Some tasks are allowed to use the machine entirely (`scaling`)

* When installing pytorch you have to make sure that it is compiled with LAPACK (for the QR decomposition)

* `mlbench-report` can be used at any time to check current results.

* Stop a run that is in progress
    * `kill -9 $(ps | grep run | awk '{print $1}' | paste -s -d ' ')`
    * `kill -9 $(ps | grep python | awk '{print $1}' | paste -s -d ' ')`

# FAQ

* When running using the AMD stack the initial compilation of each models can take a significant amount of time. You can remove the compilation step by using Mila's miopen compilation cache. To use it you can simply execute `copy_rocm_cache.sh`.

* If your machine supports SSE vector instructions you are allowed to replace it with pillow-simd for faster load times

* For machines with NUMA nodes cgroups might be set manually by the users. If the constraint below are met
    * 1 student group per GPUs (student0 for GPU 0, ... student31 for GPU 31)
    * Each student group need to be allocated an equal amount of RAM. All students should be able to use all the RAM that has been allocated to them without issues
    * Each student group need to be allocated the same amount of threads, the threads need to be mutually exclusive.

* Do all these benchmarks run/use GPUs or are some of them solely CPU-centric?
    * 2 benchmarks do not use GPUs
        * image_loader: which only measures IO speed when loading JPEG images
        * cart: which is a simplistic reinforcement learning benchmark that only uses the CPU.

* convnet and convnet_fp16 seem to be single GPU benchmarks but nvidia-smi shows activity on all GPUs in a node. Are the other GPUs used for workers?
    * They use a single GPU, but all scripts using a single GPU are launched N times in parallel where N is the number of GPU on the node.
        This is done to simulate N users running N models in parallel.

* We are using docker and `sudo` is not necessary
    * you can set `export SUDO=''` to not use sudo

* Is there a multi-node benchmark in convnets ? If yes, what was the reference run configuration ?
    * There are no multi-node benchmarks. Only Single Node Multi-GPU

* What does the cgroup script do? It looks like it is an environment-specific script and may not be relevant to our environment. Can we comment out that line and run the script?
    * No, you cannot comment out that line. The cgroups are used to emulate multiple users and force the resources of each users to be clearly segregated, similar to what Slurm does in a HPC cluster.

* While running fp16 tasks, the warnings below are shown:

        Warning:  multi_tensor_applier fused unscale kernel is unavailable, possibly because apex was installed without --cuda_ext --cpp_ext. Using Python fallback.  Original ImportError was: ModuleNotFoundError("No module named 'amp_C'",)
        Attempting to unscale a grad with type torch.cuda.HalfTensor Unscaling non-fp32 grads may indicate an error. When using Amp, you don't need to call .half() on your model.

    * Those can be ignored

# ROcm Cache

ROCm cache is structured by default like so `.cache/miopen/2.1.0/<kernel_hash>/<compiled_kernel>*.cl.o`, and
the performance database is located at `~/.config/miopen/gfx906_60.HIP.2_1_0.ufdb.txt`

We provide a zipped version of the miopen cache folder and a copy of our performance database file than you can unzip in your own cache location to speed up the first run of the benchmark.

```
unzip training/common/miopen.zip -d .cache/

cp training/common/gfx906_60.HIP.2_1_0.ufdb.txt ~/.config/miopen/
```

NB: the compile cache and performance database are both version dependent. It will only work if your version of MIOpen matches ours.

# Details

* End to End - No details; faster wins
    * Keeps track of cost value to make sure no NaNs occur
    * Synchronization every N batch to not accidentally slowdown computations
* all examples run independently on N GPUs
* Some multi GPU examples (1 to N)
* PyTorch focused
* Download dataset scripts
* fp32 and fp16 

The idea is to have one consolidated repo that can run every bench in one run 
(as opposed to MLPerf approach of everybody doing their thing).

There is a single `requirements.txt` that consolidates all the requirements of all the examples, which means the dependencies need to play nice.

-- NO DOCKER --

## Directory Layout

```
$task/$model/$framework/run*.sh...
$task/$model/download_dataset.sh
```

The run script downloads the dataset and run each run script one by one.
each script source the `config.env` before running. The file defines the location of the data sets and useful location
(temp, data, output) as well as if cuda is available, the number of devices and processors available.

Each `run*.sh` script should be runnable from any working directory

## Datasets

    du -hd 2 data/
    205M    data/wmt16/mosesdecoder
    4.5G    data/wmt16/data
    828K    data/wmt16/subword-nmt
    13G     data/wmt16
    16G     data/ImageNet/train
    16G     data/ImageNet
    73M     data/bsds500/BSR
    73M     data/bsds500
    53M     data/mnist/raw
    106M    data/mnist/MNIST
    53M     data/mnist/processed
    211M    data/mnist
    19G     data/coco/train2017
    796M    data/coco/annotations
    788M    data/coco/val2017
    20G     data/coco
    1.2M    data/time_series_prediction
    1.8G    data/ml-20m
    50G     data/

* Through Academic Torrent
  * LSUN [(168.09Gb)][1]
  * ImageNet [(166.02Gb)][2]
  * MNIST [(11.59 Mb)][3]
  * COCO [()][4]
  * ml20 [()][5]
  * wmt16 [()][6]
  * bsds500 [(70 Mb)][7]
  * SNLI
  * LibriSpeech
  

[1]: http://academictorrents.com/details/c53c374bd6de76da7fe76ed5c9e3c7c6c691c489
[2]: http://academictorrents.com/details/943977d8c96892d24237638335e481f3ccd54cfb
[3]: http://academictorrents.com/details/ce990b28668abf16480b8b906640a6cd7e3b8b21
[4]: http
[5]: http
[6]: http
[7]: http


* Fake datasets:
  * image datasets can be big some some model will generate fake images instead of downloading the real dataset


# Benchmark methodology

For each test we measure the compute time of `number` batches `repeat` times, discard the first few observations and report the average. To get the samples per second, you need to compute `batch_size / (train_time / number)`

```python
for r in range(args.repeat):

    with chrono.time('train') as t:
    
        for n in range(args.number):
            batch = next(batch_iterator)
            train(batch)
            
    print(f'[{r:3d}/{args.repeat}] ETA: {t.avg * (args.repeat - (r + 1)) / 60:6.2f} min')
```

Report output sample

```json
{
    "batch-size": 128,
    "repeat": 25,
    "number": 5,
    "train": {
            "avg": 14.0371,
            "count": 20,
            "max": 20.0015,
            "min": 11.922,
            "sd": 1.9162,
            "unit": "s"
        },
    "train_item": {
        "avg": 45.59,      // 128 * 5 / 14.037
        "max": 53.68,      // 128 * 5 / 11.922
        "min": 31.98,      // 128 * 5 / 20.015
        "range" 21.69,
        "unit": "items/sec"   // img/sec in case of Image Batch
    }
 }   
```
