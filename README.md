# ECE9143_HPML_final_project
## Goals
The first goal is to understand where the time is spent in each phases(data preprocessing time, data loading time, training time, and inference when training VGG model using CIFAR 10 dataset on CPU and GPU.

The second goal is apply three different optimization technique to boost the model and evaluate the performance in comparison to performance before the optimization.

## Usage
On NYU HPC platform, module cuda and Pytorch are needed, In the batch file, the GPU partition is RTX 8000, number of node is 1, cpu per task = 20, memory = 30GB, number of GPU for DP is 4.
To run:
```bash
python project.py --batchsize 128 --numepoch 10 --workers 2 --optimizer sgd
```
To run without gpu:
```bash
python project.py --batchsize 128 --numepoch 10 --workers 2 --optimizer sgd --no-cuda
```
To run with Data Parallel:
```bash
python project.py --batchsize 128 --numepoch 10 --workers 2 --optimizer sgd --dp
```
To run with Distributed learning:
```bash
python project.py --batchsize 128 --numepoch 10 --workers 2 --optimizer sgd --dl
```
To run with Mixed precision:
```bash
python project_mixed_precision.py --batchsize 128 --numepoch 10 --workers 2 --optimizer sgd
```
## example batch file
```bash
#!/bin/bash

#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=finalp

module purge

singularity exec --nv \
            --overlay /scratch/ps4702/lab2/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python project.py --batchsize 128 --numepoch 10 --workers 2 --optimizer sgd --dp"
```
My cuda and pytorch module was installed under lab2's directory, so my file path will look like this, but it will be different for you, as long as the modules are correctly installed, it will be fine.


## results and observations
1.performance with GPU(RTX8000) is 10 times better than running with CPU.<br />
2.Data parallel can reduce data loading time and inference time.<br />
3.Distributed learning can reduce training time and reduce inference time.<br />
4.Mixed precision significantly boost performance, result in almost half of the running time.<br />
5.Three optimization methods are all effective, but mixed precision seems to be better.<br />
