## Appendix B: Getting and Running the Benchmarks

The benchmarks are hosted on our git repository.
The benchmark is composed of 12 benchmarks, but they are all grouped in a single script for you to run everything in one shot.
The benchmarks require to download data from the internet so the compute node will need an internet access.


## Setting up [Bare Bone]

Before running the benchmark the environment needs to be setup.

* Fetch the benchmarks
    * `git clone https://github.com/mila-iqia/training.git`
    * `git checkout -b vendor-name`
    * `cd training`
    * [Optional]
        * `virtualenv mlperf --python=python3.6`
        * `source ~/mlperf/bin/activate`
* Setup the base path to be used in the benchmark to download & store generated data
     * `export BASE=~/mlperf/`
* Create the cgroup that is going to be used to run the different benchmarks
    * `sudo ./cgroup_setup.sh`
*Install the dependencies
    * The benchmarks are using pytorch; it should be installed before hands.
    Given the diversity of plateforms available, we do not provide instructions for pytorch
    Hardware vendors should have containers or instructions that you can use
        * NVIDIA: NGC containers
        * AMD: ROCm containers
        * IBM: PowerAI containers
    * `sudo apt install $(cat apt_packages)`
    * `pip install -e common`
    * `pip install Cython`
    * `pip install --no-deps -r requirements.txt`
* Download the datasets
    * `./download_datasets.sh`

## Running the Benchmarks

* Baseline submission [Required]
    * `cp baselines.json vendor-name_base.json`
        Set --batch-size, --seed, --num-processes,  --workers, --cuda to appropriate values for your device
    * `./run.sh --jobs vendor-name_base.json
        *  If the run script fails or you want to rerun a specific benchmark. It can be run alone using `run.sh --name reso`. 
        * The name of the benchmark can be found in the configuration file
    * `git add --all`
    * `git commit -m "vendor-name baseline submission"`
    * `git push origin vendor-name-base`
       
* Tweaked submission [Optional]
    * `git checkout -b vendor-name-tweak`
    * Tweak the code 
    * `./run.sh --jobs vendor-name_base.json`
    * `git add --all && git commit -m "vendor-name baseline submission" && git push origin vendor-name-tweak`

* Zip the directory `$BASE/output/bench_results` and send it to us. Where $BASE is the directory you specified during setup


### Details

The script will run 12 benchmarks with different configurations.
The configurations are stored inside the `baselines.json` file.
You can find below an example of a bench configuration.
The examples specify the `reso` benchmark that will be ran 2 times with a batch size of 32 and then 64 using the cgroup `student`

    {
        "name": "reso",
        "cmd": "./super_resolution/subpixel_convolution/pytorch/run.sh",
        "args":{
          "--upscale_factor": 2,
          "--batch-size": [32, 64],
          "--testBatchSize": 8,
          "--repeat": 50,
          "--number": 20,
          "--workers": 8,
          "--seed": 0,
          "--lr": 0.1,
          "--no-checks": "",
          "--report": "$OUTPUT_DIRECTORY/$JOB_FILE",
          "--cuda": ""
        },
        "cgroup": "student"
    },

Benchmarks include:

* Single GPU training
* Multi GPU training
* FP16 and FP32 training

To simulate multiple user on a single machine each single GPU benchmark will be launched N times.
N representing the number of GPUs the machine holds.

When a single GPU is used only `(NPROC / NGPU)` threads and `(MEM / NGPU)` memory are available to the python process.
Where `NPROC` is the total number of threads, `NGPU` is this total number of GPU and `MEM` is the total amount of RAM of the entire system.

Greater importance is given to single GPU & FP32 training, the benchmark reflect that.
Every time a benchmark is ran it will generate one report per GPU in a json format.
You can find below an example of output.

We keep track the batch loss just to make sure the accelerators computation are consistent and
that NaNs do not popups where it should not. The important metric that we are looking at will be "train_item"
In a vision context this number represent the number of image per second that the GPU is processing.

The version tag is important. It is the hash mismatch with ours we will not be able to validate your results.
If modifications to the script needs to be made those needs to be disclosed to us.

    {
        "batch_loss": [
            1.6796641855165717e-10,
            1.7485513037485134e-10,
            2.937983190065552e-10,
            3.277212945462793e-10,
        ],
        "batch_size": 256,
        "cpu_cores": 10,
        "cuda": true,
        "devices": 1,
        "epoch_loss": [],
        "gpu": "TITAN RTX",
        "hostname": "RTX1",
        "jr_id": "0",
        "metrics": {},
        "name": "regression_polynome_pytorch_main.py",
        "number": 1000,
        "repeat": 20,
        "train": {
            "avg": 87.82592058181763,
            "count": 10,
            "max": 90.07928371429443,
            "min": 84.95557308197021,
            "sd": 1.5483211029291235,
            "unit": "s"
        },
        "train_item": {
            "avg": 291.48570069528994,
            "max": 301.33396870031813,
            "min": 284.194.0893002195,
            "range": 17.139879400098638,
            "unit": "items/sec"
        },
        "unique_id": "4c9a6df0b7c25376f7f048882857ea58ac272cc81a2dfc1b403c0d8c908dc70f",
        "vcd": "0",
        "version": "308a5984e9",
        "workers": 0
    }

## Methodology

We measure the compute time of number batches repeat times, discard the first few observations and report the average.
To get the samples per second you need to compute batch_size / (train_time / number)
You can find a example below.

    for r in range(args.repeat):
    
        with chrono.time('train') as t:
        
            for n in range(args.number):
                batch = next(batch_iterator)
                train(batch)
                
        print(f'[{r:3d}/{args.repeat}] ETA: {t.avg * (args.repeat - (r + 1)) / 60:6.2f} min')


## Additional Guidelines

We allow two benchmark submissions. The first one is **baseline** submission. It consist of running the benchmark suite
with minimal modification. The second is the **tweaked** submission in which you are allowed to modify the code to 
optimize runtime.

* **baseline**
    * The vendors can fine tune their device to the **baseline** benchmark using ONLY the `vendor-name_base.json` configuration file.
    * Only model a fixed set of arguments in the configuration file can be modified.
    * Arguments that can be modified are:
        * `--batch-size`, `--num-processes`, `--workers` and `--cuda`
    * Modifications of the source code is only permissible to enable correct execution of the benchmark
        * We reserve the right to refuse the modification if we judge them to be too extensive  
    * The tuned configuration file must be uploaded to the git repository 

* **Tweaked**
    * Source code can be modified but the modified version needs to be uploaded to the git repository.


1. ...


## Result Grid

* GPU reports are averaged out.
* A report will be selected as a baseline to get relative results instead of absolute measures
* Benchmark will be averaged out together 



A baseline report `A` will be picked from the submitted files.
The other submission results will be divided by the results of `A` to normalize the results.
`A` will be 1.



 




