Training Benchmarks
===================

# How to run it

## Bare bone

* Tested on **python 3.6**

* Install dependencies
```bash
$ sudo apt install git
$ git clone https://github.com/mila-iqia/training.git
$ cd training
$ sudo apt install $(cat apt_packages)

$ virtualenv ~/mlperf --python=python3.6
$ source activate ~/mlperf/bin/activate

$ python --version
> Python 3.6.4

$ pip install -e common
$ pip instal Cython
$ pip install numpy
$ pip install --no-deps -r requirements.txt

$ export BASE=~/data/
$ ./cgroup_setup.sh
$ ./download_datasets.sh
```

* Execute the benchmarks

```batch
$ source activate ~/mlperf/bin/activate
$ export BASE=~/data/
$ ./run.sh --jobs baselines.json

$ cp baselines.json tweaked.json

# modify tweaked.json to reflect the device capacity
$ ./run.sh --jobs tweaked.json  # run the tweaked version
```

* To get reproducible results we recommend the user to run the benchmark 10 times using `./run_10.sh`
After running the benchmarks 10 times you can use `mlbench-report` to get a report for your device.
The tool requires a minimum of 4 runs to work.

```bash
$ mlbench-report --reports $BASE/ --name baselines  # <= baselines if the name of the baseline report
$ mlbench-report --reports $BASE/ --name tweaked    

                                                                                         result           sd       sd%
atari_e2f3fe0546_dd03cd0f4d73da1849dda236a888941114b38d344a6ebea26da5712cd129...       1.239247     0.022964  0.018530
cart_3c703de83b_ccba6b093b526f0cb391038014620a27ac5d84c9f9f2a7dc18a1be765e416eae    3408.961050    72.073918  0.021142
convnet_8661315414_32e210092e9f466417b919b8c28fb43846937497273a57cf2811fd95b4...     143.165105     0.681476  0.004760
convnet_distributed_c6bba3b4f4_6b07c0ce75d6f14f5885285777ef6955755490b30e6375...     225.988936     1.025696  0.004539
convnet_distributed_fp16_c6bba3b4f4_c6fafa7fa5f9039189fca8f36891cc6ce34bd8433...     315.806405     0.845454  0.002677
convnet_fp16_8661315414_679f3aebe6899e635d2d4f6b55892408a5162f7943c3c2029e0b1...     188.526259     0.411091  0.002181
dcgan_all_b9e0573060_2001e4a6ae67bd21141ab6245d730e4f4a8015d45c024d4c9fa1b174...     261.599951     1.302347  0.004978
dcgan_b9e0573060_4479b0a0da45f9047203abc12bb3f8ad77ad2f7eaabdda5bedfc8da339cf...     192.102628     4.167030  0.021692
fast_style_8548745942_4a7c164fce0e0eeccf6342775301273d57213a82ab1ee72257aa14e...     179.432926     1.161558  0.006473
loader_89452a8721_6d9e8d8f168b5aa079a7e07060bfd85117230c64b3e7d6b16ac7cb34383...    1328.147562   120.488680  0.090719
recom_4e5416d201_32a02f06e63a4d708008ece4eadff6ea41e80c1926a6c0a389b5fe05d31e...   18683.003248   140.090284  0.007498
reso_2db952ad83_4eb5e9316440d9e22a9b8dd02d1af87f81850581837fec3a0a34a8ddac559a8d     152.090354     1.058333  0.006959
ssd_c861f831bf_b19c65f55694a1ce268db6a506576daaf10e252581ecfb6b0cb0631205a41011       58.581525     0.461788  0.007883
toy_lstm_c91749e5dc_f1e2cdca8961ea9f71e8acdbf7d256cace772e3622618a10b068189e2...       0.479538     0.005167  0.010776
toy_reg_308a5984e9_e8dc1b0a9a06bb2a0719a09869a9737049edfa893e353572d116d31411...  147298.734937  3778.620055  0.025653
translator_e89027e02b_e5a9a2e97e7d140dfe0f8093939c58c4ce6486b96f9a53ec9a3f3f5...      86.263107     0.333906  0.003871
vae_045f044441_cd5c0c8d9c6dd51e41753854900087feb2cbff729c67380f72daf2df1982e3b7     4717.010770    56.374548  0.011951
wlm_2c81324f51_943d4e0dc55a9d8a6ae22caaaee35710fd8cc13a304f32b03644f4fb5279de3e      705.432753     5.092593  0.007219
wlmfp16_9db33cdeec_5458c14e5aaef1c97c33b35231ae717d0bc5ebc522d7cedcddad166f24...    1282.657099     5.394817  0.004206
--
Error Margin: 1.3879 %
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
> cd signularity
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

```
./regression/polynome/pytorch/run.sh     0.45 min passed
./time_sequence_prediction/lstm/pytorch/run.sh     1.79 min passed
./variational_auto_encoder/auto_encoding_variational_bayes/pytorch/run.sh     1.12 min passed
./image_loading/loader/pytorch/run.sh     3.21 min passed
./super_resolution/subpixel_convolution/pytorch/run.sh     4.20 min passed
./natural_language_processing/rnn_translator/pytorch/run.sh     4.69 min passed
./natural_language_processing/word_language_model/pytorch/run.sh     2.14 min passed
./natural_language_processing/word_language_model/pytorch/run_fp16.sh     1.79 min passed
./reinforcement/cart_pole/pytorch/run.sh     1.97 min passed
./reinforcement/atari/pytorch/run.sh     6.46 min passed
./object_detection/single_stage_detector/pytorch/run.sh    16.76 min passed
./object_detection/single_stage_detector/pytorch/run.sh     9.80 min passed
./fast_neural_style/neural_style/pytorch/run.sh     4.55 min passed
./generative_adversarial_networks/dcgan/pytorch/run.sh     4.72 min passed
./image_classification/convnets/pytorch/run.sh     1.17 min passed
./image_classification/convnets/pytorch/run.sh     0.96 min passed
./recommendation/neural_collaborative_filtering/pytorch/run.sh    13.08 min passed
./image_classification/convnets/pytorch/run_distributed.sh     1.69 min passed
./image_classification/convnets/pytorch/run_distributed.sh     2.34 min passed
./generative_adversarial_networks/dcgan/pytorch/run.sh     3.91 min passed
Total Time  6332.38 s
```

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
