Training Benchmarks
===================

# How to run it

## Barebone

* Tested on **python 3.6** with **pytorch 1.3**

### Anaconda

```bash
$ sudo apt install git
$ git clone https://github.com/mila-iqia/training.git
$ cd training

$ ./install_dependencies.sh
# reload bash with anaconda
$ exec bash
$ conda activate mlperf
$ pip install --no-deps -r requirements.txt

# Install pytorch
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

* Execute the benchmarks

```batch
$ conda activate mlperf
$ export BASE=~/data/

$ ./cgroup_setup.sh
$ ./download_datasets.sh
$ ./run.sh --jobs fast.json

$ cp baselines.json tweaked.json

# modify tweaked.json to reflect the device capacity
$ ./run.sh --jobs tweaked.json  # run the tweaked version
```

### VirtualEnv

* Install dependencies
```bash
$ sudo apt install git
$ git clone https://github.com/mila-iqia/training.git
$ cd training
$ sudo apt update
$ sudo apt install $(cat apt_packages)

$ virtualenv ~/mlperf --python=python3.6
$ source ~/mlperf/bin/activate

$ python --version
> Python 3.6.4

$ pip install -e milabench
$ pip install Cython
$ pip install numpy
$ pip install --no-deps -r requirements.txt
$
$ > Install Pytorch Now
$
$ export BASE=~/data/
$ ./cgroup_setup.sh
$ ./download_datasets.sh
```

* Execute the benchmarks

```batch
$ source activate ~/mlperf/bin/activate
$ export BASE=~/data/

$ ./cgroup_setup.sh
$ ./download_datasets.sh
$ ./run.sh --jobs fast.json

$ cp baselines.json tweaked.json

# modify tweaked.json to reflect the device capacity
$ ./run.sh --jobs tweaked.json  # run the tweaked version
```

* To get reproducible results we recommend the user to run the benchmark 10 times using `./run_10.sh`
After running the benchmarks 10 times you can use `mlbench-report` to get a report for your device.
The tool requires a minimum of 4 runs to work.

```bash
$ mlbench-report --reports $BASE/ --name baselines --gpu-model MI50 # <= baselines if the name of the baseline report
$ mlbench-report --reports $BASE/ --name fast   --gpu-model RTX
                             target         result           sd       sd%      diff
bench                                                                              
atari                          5.41       7.424253     0.055735  0.007507  0.372320
cart                        2000.00    2195.248475     6.457116  0.002941  0.097624
convnet                      178.76     175.973774     0.191682  0.001089 -0.015586
convnet_distributed          276.10     756.841070    60.333975  0.079718  1.741185
convnet_distributed_fp16     330.41     785.470579    43.330802  0.055165  1.377260
convnet_fp16                 262.56     329.347550     0.770485  0.002339  0.254371
dcgan_all                    139.29     274.608760     1.361280  0.004957  0.971489
dcgan                        115.52     147.794385     0.632454  0.004279  0.279384
fast_style                   156.75     146.285007     0.218744  0.001495 -0.066762
loader                      1200.00    1188.224303    62.070805  0.052238 -0.009813
recom                      20650.19   20802.705368   103.476919  0.004974  0.007386
reso                         138.34     178.694817     0.920870  0.005153  0.291708
ssd                           59.96      49.263719     0.203476  0.004130 -0.178390
toy_lstm                       2.65       1.254635     0.007228  0.005761 -0.526553
toy_reg                   368820.18  215482.487073  3745.776792  0.017383 -0.415752
translator                   223.47     212.040509     0.443492  0.002092 -0.051146
vae                         7972.69    5931.232565    26.450238  0.004459 -0.256056
wlm                         1440.87    1365.497700     2.910483  0.002131 -0.052310
wlmfp16                     3793.37    4649.460342    18.698980  0.004022  0.225681
--

Statistics               |     Value | Pass |
-------------------------|-----------|------|
Bench Passes             :           | True
Deviation Quantile (80%) : +1.1458 % | True |
Performance              : +0.2129 % | True 
--
```

## Singularity [Experimental]

3 Containers are experimentally supported
* ROCm: for AMD GPUs (x86_64)
* PowerAI for IBM Power9 (ppe_64le) + (with NVIDIA GPUs)
* NGC for NVIDIA GPUs on (x86_64)

Some prebuilt container can be found at [training-container][100]

[100]: https://www.singularity-hub.org/collections/3109


```bash
> sudo apt install git

> git clone https://github.com/mila-iqia/training.git
> cd training

# Install & Generate the Singularity files
> ./install_singularity.sh
> cd singularity
> python generate_container.py
> singularity build rocm.simg Singularity.rocm
> cd ..

# Run the benchmarks
> export BASE=~/location
> ./run.sh --singularity rocm.simg [--jobs baselines.json]
```


## Docker [Experimental]

You can use cgroups and docker using the script below.

```bash
$ sudo docker run --cap-add=SYS_ADMIN --security-opt=apparmor:unconfined -it my_docker
$ apt-get install cgroup-bin cgroup-lite libcgroup1
$ mount -t tmpfs cgroup_root /sys/fs/cgroup

$ mkdir /sys/fs/cgroup/cpuset
$ mount -t cgroup cpuset -o cpuset /sys/fs/cgroup/cpuset

$ mkdir /sys/fs/cgroup/memory
$ mount -t cgroup memory -o memory /sys/fs/cgroup/memory
```

## Details

You can run individual test using the command below

```bash 
./run.sh --jobs baselines.json --name vae
```

* the benchmark starts with two toy examples to make sure everything is setup properly

* Each bench run `N_GPU` times in parallel with only `N_CPU / N_GPU` and `RAM / N_GPU` to simulate multiple users.
  * if your machine has 16 GPUs and 32 cores, the bench will run in parallel 16 times with only 2 cores for each GPUs.

* Some tasks are allowed to use the machine entirely (`convnet_all`, `dcgan_all`)

* When installing pytorch you have to make sure that it is compiled with LAPACK (for the QR decomposition)
   
* Since the tests run for approximately 3h you can check the result of each step by doing `cat $OUTPUT_DIRECTORY/summary.txt`

* Stop a run that is in progress ?
    * `kill -9 $(ps | grep run | awk '{print $1}' | paste -s -d ' ')`
    * `kill -9 $(ps | grep python | awk '{print $1}' | paste -s -d ' ')`
    
* The outputs are located at `$BASE/output`.

* You can check overall status by looking at `cat $BASE/output/summary.txt `

```bash
$ cat perf/output1/summary.txt 
[ 1/19][  /  ] PASSED |     0.27 MIN | ./regression/polynome/pytorch/run.sh
[ 2/19][  /  ] PASSED |     0.93 MIN | ./time_sequence_prediction/lstm/pytorch/run.sh
[ 3/19][  /  ] PASSED |     0.78 MIN | ./variational_auto_encoder/auto_encoding_variational_bayes/pytorch/run.sh
[ 4/19][  /  ] PASSED |     2.50 MIN | ./image_loading/loader/pytorch/run.sh
[ 5/19][  /  ] PASSED |     1.26 MIN | ./super_resolution/subpixel_convolution/pytorch/run.sh
[ 6/19][  /  ] PASSED |     3.20 MIN | ./natural_language_processing/rnn_translator/pytorch/run.sh
[ 7/19][  /  ] PASSED |     1.25 MIN | ./natural_language_processing/word_language_model/pytorch/run.sh
[ 8/19][  /  ] PASSED |     0.95 MIN | ./natural_language_processing/word_language_model/pytorch/run_fp16.sh
[ 9/19][  /  ] PASSED |     0.91 MIN | ./reinforcement/cart_pole/pytorch/run.sh
[10/19][  /  ] PASSED |     2.98 MIN | ./reinforcement/atari/pytorch/run.sh
[11/19][  /  ] PASSED |     3.95 MIN | ./object_detection/single_stage_detector/pytorch/run.sh
[12/19][  /  ] PASSED |     2.13 MIN | ./fast_neural_style/neural_style/pytorch/run.sh
[13/19][  /  ] PASSED |     2.08 MIN | ./generative_adversarial_networks/dcgan/pytorch/run.sh
[14/19][  /  ] PASSED |     0.85 MIN | ./image_classification/convnets/pytorch/run.sh
[15/19][  /  ] PASSED |     0.91 MIN | ./image_classification/convnets/pytorch/run.sh
[16/19][  /  ] PASSED |     9.34 MIN | ./recommendation/neural_collaborative_filtering/pytorch/run.sh
[17/19][  /  ] PASSED |     1.79 MIN | ./image_classification/convnets/pytorch/run_distributed.sh
[18/19][  /  ] PASSED |     2.42 MIN | ./image_classification/convnets/pytorch/run_distributed.sh
[19/19][  /  ] PASSED |     8.48 MIN | ./generative_adversarial_networks/dcgan/pytorch/run.sh
Total Time  2817.82 s
```

# FAQ

* When running using the AMD stack the initial compilation of each models can take a significant amount of time.
You can remove the compilation step by using Mila's miopen compilation cache. 
To use it you can simply execute `copy_rocm_cache.sh`.

* If your machine supports SSE vector instructions you are allowed to replace it with pillow-simd for faster load times

* For machines with NUMA nodes cgroups might be set manually by the users. If the constraint below are met
    * 1 student group per GPUs (student0 for GPU 0, ... student31 for GPU 31)
    * Each student group need to be allocated an equal amount of RAM. All students should be able to use all the RAM that has been allocated to them without issues
    * Each student group need to be allocated the same amount of threads, the threads need to be mutually exclusive.

* Do all these benchmarks run/use GPUs or are some of them solely CPU-centric?
    * 2 benchmarks do not use GPUs
        * image_loader: which only measures IO speed when loading JPEG Images
        * cart: which is a simplistic RL bench that only uses the CPU.
        
* convnet and convnet_fp16 seem to be single GPU benchmarks but nvidia-smi shows activity on all GPUs in a node. Are the other GPUs used for workers?
    * They use a single GPU, but all scripts using a single GPU are launched N times in parallel where N is the number of GPU on the node.
        This is done to simulate N users running N models in parallel.

* Does the --workers argument launch 1 worker thread per process? Eg. In dcgan_all, where --workers=8 and --ngpu=$DEVICE_TOTAL, for 8 GPUs, will this launch 8 workers or 64 workers?
    * The `--workers W` argument is used to initialize python dataloader which will spawn W child processes / workers to load W batches in parallel.
    In the case of dcgan_all --workers=8 will launch 8 workers for the 8 GPUs because it uses pytorch dataparallel to split and then execute a batch on multiple GPUs.

* We are using docker and `sudo` is not necessary
    * you can set `export SUDO=''` to not use sudo
    
* Is there a multi-node benchmark in convnets ? If yes, what was the reference run configuration ?
    * There are no multi-node benchmarks. Only Single Node Multi-GPU
    
* What does the cgroup script do ? It looks like it is environmental specific script and may not be relevant to our environment. Can we comment out that line and run the script ?
    * No, you cannot comment out that line. The cgroups are used to emulate multiple users and force the 
    resources of each users to be clearly segregated, similar to what Slurm 
    does in a HPC cluster.

* While running fp16 task versions the warning below are shown

        Warning:  multi_tensor_applier fused unscale kernel is unavailable, possibly because apex was installed without --cuda_ext --cpp_ext. Using Python fallback.  Original ImportError was: ModuleNotFoundError("No module named 'amp_C'",)
        Attempting to unscale a grad with type torch.cuda.HalfTensor Unscaling non-fp32 grads may indicate an error. When using Amp, you don't need to call .half() on your model.

    * Those can be ignored

* The report show duplicated benchmarks:
    * It means you modified the script or the arguments and the version tag is different

# ROcm Cache

ROCm cache is structured by default like so `.cache/miopen/2.1.0/<kernel_hash>/<compiled_kernel>*.cl.o`, and
the performance database is located at `~/.config/miopen/gfx906_60.HIP.2_1_0.ufdb.txt`

We provide a zipped version of the miopen cache folder and a copy of our performance database file than you can unzip in your own cache location to speed up the first run of the benchmark.

```
unzip training/common/miopen.zip -d .cache/

cp training/common/gfx906_60.HIP.2_1_0.ufdb.txt ~/.config/miopen/
```

NB: the compile cache and performance database are both version dependent. It will only work if your version of MIOpen matches ours.

# Features

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

There is a single `requirements.txt` that consolidate all the requirements of all the examples.
Which means the dependencies need to play nice.

-- NO DOCKER --

# Directory Layout

```
$task/$model/$framework/run*.sh...
$task/$model/download_dataset.sh
```

The run script downloads the dataset and run each run script one by one.
each script source the `config.env` before running. The file defines the location of the data sets and useful location
(temp, data, output) as well as if cuda is available, the number of devices and processors available.

Each `run*.sh` script should be runnable from any working directory

# Data Sets

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

Measure the compute time of `number` batches `repeat` times, discard the first few observations and report the average.
To get the samples per second you need to compute `batch_size / (train_time / number)`

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

 image/sec = 128/(14.0371/5)
 image/sec = 45.59
```

# Example

## Stdout example


    --------------------------------------------------------------------------------
                        batch_size: 32
                              cuda: True
                           workers: 4
                              seed: 0
                           devices: 1
                             jr_id: 0
                               vcd: 0
                         cpu_cores: 0
                              data: /media/setepenre/UserData/tmp/fake/
                              arch: resnet18
                                lr: 0.1
                         opt_level: O0
                            repeat: 15
                            number: 1
                            report: None
    --------------------------------------------------------------------------------
    [  0/ 15] | ETA:   0.87 min | Batch Loss   7.0346
    [  1/ 15] | ETA:   1.20 min | Batch Loss   6.3855
    [  2/ 15] | ETA:   1.32 min | Batch Loss   5.8987
    [  3/ 15] | ETA:   1.34 min | Batch Loss   6.1524
    [  4/ 15] | ETA:   1.18 min | Batch Loss   5.7566
    [  5/ 15] | ETA:   1.09 min | Batch Loss   5.3836
    [  6/ 15] | ETA:   0.94 min | Batch Loss   5.7247
    [  7/ 15] | ETA:   0.81 min | Batch Loss   5.1463
    [  8/ 15] | ETA:   0.69 min | Batch Loss   5.2389
    [  9/ 15] | ETA:   0.59 min | Batch Loss   5.3700
    [ 10/ 15] | ETA:   0.47 min | Batch Loss   5.4471
    [ 11/ 15] | ETA:   0.36 min | Batch Loss   5.4479
    [ 12/ 15] | ETA:   0.24 min | Batch Loss   5.4052
    [ 13/ 15] | ETA:   0.12 min | Batch Loss   5.2305
    [ 14/ 15] | ETA:   0.00 min | Batch Loss   5.3205
    --------------------------------------------------------------------------------
    {
        "arch": "resnet18",
        "batch_loss": [
            6.9410552978515625,
            7.034599304199219,
            6.076425075531006,
            6.385458946228027,
                ...
            5.21469783782959,
            5.447901725769043,
            5.630765914916992,
            5.405197620391846,
            4.737128257751465,
            5.230467319488525,
            5.597040176391602,
            5.3204569816589355
        ],
        "batch_size": 32,
        "cpu_cores": 32,
        "cuda": true,
        "data": "/media/setepenre/UserData/tmp/fake/",
        "devices": 1,
        "epoch_loss": [],
        "gpu": "GeForce GTX 1060 6GB",
        "hostname": "Midgard",
        "jr_id": 0,
        "lr": 0.1,
        "metrics": {},
        "name": "image_classification_convnets_pytorch_conv_simple.py",
        "number": 1,
        "opt_level": "O0",
        "repeat": 15,
        "train": {
            "avg": 7.1848473072052,
            "count": 5,
            "max": 7.299542665481567,
            "min": 7.058995008468628,
            "sd": 0.09646150347927232,
            "unit": "s"
        },
        "train_item": {
            "avg": 4.4538176848809785,
            "max": 4.533223208347621,
            "min": 4.383836284884416,
            "range": 0.14938692346320526,
            "unit": "items/sec"
        },
        "unique_id": "8f3b6f5a72105215fde07fac915cad0ab83645271fc3bb812fca1b0420cdff28",
        "vcd": 0,
        "version": "18ac3cd560",
        "workers": 4
    }
    --------------------------------------------------------------------------------


## Report example

One report per GPU is generated.

```
[{
    "arch": "resnet18",
    "batch_loss": [
        6.9410552978515625,
        7.034599304199219,
        6.551750183105469,
        6.381999969482422,
        6.146703243255615,
        6.265162467956543,
        5.999067306518555,
        5.874876976013184,
        5.6117730140686035,
        6.40594482421875,
        5.765414237976074,
        5.849427700042725,
        5.572443008422852,
        5.235463619232178,
        5.402827739715576,
        5.505850791931152,
        5.451850414276123,
        5.378958702087402,
        5.547319412231445,
        5.166872978210449,
        5.242769718170166,
        5.215421676635742,
        5.263228416442871,
        5.569464206695557,
        5.162929058074951,
        5.478224277496338,
        4.759178638458252,
        5.287482261657715,
        5.222555160522461,
        5.160109519958496,
        5.05403470993042,
        4.897465229034424,
        5.299546718597412,
        5.019275665283203,
        5.1086626052856445,
        5.117300510406494,
        5.066164970397949,
        5.152365207672119,
        4.924874305725098,
        5.233489036560059,
        4.916210651397705,
        5.396081924438477,
        5.081722736358643,
        4.814561367034912,
        5.420909404754639,
        5.400428295135498,
        5.078923225402832,
        5.0143513679504395,
        5.273678779602051,
        5.099527359008789,
        5.070191383361816,
        5.1728057861328125,
        4.894495964050293,
        5.25480318069458,
        4.910001754760742,
        5.117621421813965,
        5.08811092376709,
        5.172547817230225,
        5.181445598602295,
        4.937402725219727,
        5.481322288513184,
        5.01015043258667,
        5.09744930267334,
        5.013396263122559,
        5.033565998077393,
        5.085068702697754,
        4.980317115783691,
        4.845536231994629,
        5.229896545410156,
        5.113539218902588,
        4.976269245147705,
        5.087772846221924,
        5.053828239440918,
        5.224181652069092,
        5.057964324951172,
        4.860743045806885,
        5.29210090637207,
        4.910505294799805,
        5.0234270095825195,
        5.342507362365723,
        5.017017841339111,
        5.039641380310059,
        5.222891330718994,
        5.031618118286133,
        5.141876220703125,
        5.186580181121826,
        5.074064254760742,
        5.088891506195068,
        5.131050109863281,
        4.939560413360596
    ],
    "batch_size": 32,
    "cpu_cores": 32,
    "cuda": true,
    "data": "/media/setepenre/UserData/tmp/fake/",
    "devices": 1,
    "epoch_loss": [],
    "gpu": "GeForce GTX 1060 6GB",
    "hostname": "Midgard",
    "jr_id": 0,
    "lr": 0.1,
    "metrics": {},
    "name": "image_classification_convnets_pytorch_conv_simple.py",
    "number": 5,
    "opt_level": "O0",
    "repeat": 15,
    "train": {
        "avg": 10.032922506332397,
        "count": 5,
        "max": 10.155806064605713,
        "min": 9.798663139343262,
        "sd": 0.09637364711840406,
        "unit": "s"
    },
    "train_item": {
        "avg": 15.947496843418666,
        "max": 16.328758089210496,
        "min": 15.754534793414432,
        "range": 0.5742232957960631,
        "unit": "items/sec"
    },
    "unique_id": "966c9e2984121be717c7b4f4d71637619a59b69f84a26aed8d2ea34a08e81157",
    "vcd": 0,
    "version": "18ac3cd560",
    "workers": 4
},{
    "arch": "resnet18",
    "batch_loss": [
        6.9410552978515625,
        7.034599304199219,
        6.076425075531006,
        6.385458946228027,
        6.320394515991211,
        5.8986897468566895,
        6.044133186340332,
        6.152379035949707,
        5.988056182861328,
        5.7566118240356445,
        5.904807090759277,
        5.3836259841918945,
        5.549648284912109,
        5.7246623039245605,
        5.412194728851318,
        5.146271228790283,
        5.570966720581055,
        5.238946437835693,
        5.1303300857543945,
        5.369985580444336,
        5.195217132568359,
        5.447091102600098,
        5.21469783782959,
        5.447901725769043,
        5.630765914916992,
        5.405197620391846,
        4.737128257751465,
        5.230467319488525,
        5.597040176391602,
        5.3204569816589355
    ],
    "batch_size": 32,
    "cpu_cores": 32,
    "cuda": true,
    "data": "/media/setepenre/UserData/tmp/fake/",
    "devices": 1,
    "epoch_loss": [],
    "gpu": "GeForce GTX 1060 6GB",
    "hostname": "Midgard",
    "jr_id": 0,
    "lr": 0.1,
    "metrics": {},
    "name": "image_classification_convnets_pytorch_conv_simple.py",
    "number": 1,
    "opt_level": "O0",
    "repeat": 15,
    "train": {
        "avg": 7.1848473072052,
        "count": 5,
        "max": 7.299542665481567,
        "min": 7.058995008468628,
        "sd": 0.09646150347927232,
        "unit": "s"
    },
    "train_item": {
        "avg": 4.4538176848809785,
        "max": 4.533223208347621,
        "min": 4.383836284884416,
        "range": 0.14938692346320526,
        "unit": "items/sec"
    },
    "unique_id": "8f3b6f5a72105215fde07fac915cad0ab83645271fc3bb812fca1b0420cdff28",
    "vcd": 0,
    "version": "18ac3cd560",
    "workers": 4
},
```
