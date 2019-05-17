Training Benchmarks
===================

# How to run it

```bash
git clone ....
cd training
git checkout vendor

sudo apt install $(cat apt-packages)

./cgroup_setup.sh

# install dependencies
pip install --no-deps -r  requirements.txt
cd common
python setup.py install
cd ..

# install pytorch
...

# This will run the baselines bench
#  ~ 4-9h depending on hardware

export BASE=/home/mila/mlperf
./run.sh [--jobs baselines.json]
cp $BASE/output/baselines*.json $BASE/output/results/

# Tweak the bench for better perf
cp baselines.json vendor.json
vi vendor.json

./run.sh --jobs vendor.json
cp $BASE/output/vendor*.json $BASE/output/results/

# Push your change
git add --all
git commit -m "vendor tweaked"
git push

cd $BASE/output/
zip results
send results.zip

```

You can run individual test using the command below

```bash 
./run.sh -jobs baselines.json --name vae
```

* the benchmark starts with two toy examples to make sure everything is setup properly

* Each bench run `N_GPU` times in parallel with only `N_CPU / N_GPU` and `RAM / N_GPU` to simulate multiple users.
  * if your machine has 16 GPUs and 32 cores, the bench will run in parallel 16 times with only 2 cores.

* Some tasks are allowed to use the machine entirely (`convnet_all`, `dcgan_all`)

* When installing pytorch you have to make sure that it is compiled with LAPACK (for the QR decomposition)

* We leave it as an option to provide tweaked numbers as well. 
    * Obviously not all tweaks are equals, 
    you can find below a list from the most desirable kind of tweaks to the least desirable. 
    NB: Some models have configurable hidden layers, obviously modifying those values are not valid optimizations
    NB2: Not all models are worth optimizing
    NB3: Not all configuration will work but at least one should
    NB4: Some code change might be required for specific vendor 
    
        1. argument change (batch size, worker/process count, etc..) (no code change)
        2. environment i.e execution in an optimized container (no code change)
        3. code tweak
   
* Since the tests run for approximately 9h you can check the result of each step by doing `cat $OUTPUT_DIRECTORY/summary.txt`

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


# Options

FAKE_DATASET: generate a pseudo dataset instead of downloading the original
USE_VALIDATION: use validation set to train (do not download the training dataset)
    
# Test Breakdown

# Singularity

## Make the Singularity Image

```bash
 sudo singularity build --sandbox vendor_img/ vendor.sif
 sudo singularity shell --writable vendor_img
 mkdir /mila
 chmod 777 /mila
 cd /mila
 cat requirements.txt | xargs pip install
 cd common 
 python setup.py install
```

## Use the image

```bash
singularity shell -B /storage/:/data/ vendor_img
cd /mila
./run.sh
```


# Submitting Tuned Results

We allow tuning on the vendor'side.


1. Make sure FS cache is empty
    * To ease up the benchmark process we only run on a reduce dataset which means that most if not all the dataset fit
    in RAM. This is not something that we want to measure as it will skew the process.
    You should make sure that the caches are clean before running the test.
    When validating your results we will
    
```bash
    sh -c 'echo 1 > /proc/sys/vm/drop_caches'
    sh -c 'echo 2 > /proc/sys/vm/drop_caches'
    sh -c 'echo 3 > /proc/sys/vm/drop_caches'
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
    3.275667142868042
    --------------------------------------------------------------------------------
    {
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
    }
    --------------------------------------------------------------------------------


## Report example

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
