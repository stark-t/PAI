# Overview

This readme file walks you through setting the environments for running the training scripts on a Linux cluster that offers nodes with multiple GPUs. Each training was run on a single node. 

Computations were done using resources of the Leipzig University Computing Centre.

# Clone PAI repository

Clone our PAI repository in the home directory of the landing node:
```bash
pwd
# Should see something like /home/sc.uni-leipzig.de/your_user_name

git clone https://github.com/stark-t/PAI.git # if you put .git, will require authentication (can also skip .git)
```

# YOLOv5 - prepare dependencies and environment

Clone the repository of YOLOv5 in `.../PAI/detectors`:
```bash
cd ~/PAI/detectors/
git clone https://github.com/ultralytics/yolov5
```

Create an environment with the needed dependecies/requirements for YOLOv5:

```bash
cd ~ # move back to home

# Start from a clean global environment
module purge 

# Load most up to date Python version on the cluster
module load Python/3.9.6-GCCcore-11.2.0

# Create a virtual environment named yolov5 in ~/venv
python -m venv ~/venv/yolov5
# If you need to delete the environment and start from scratch, then use rm -r ~/venv/yolov5
# Then use the command from above to create the environment

# Activate virtual environment
source ~/venv/yolov5/bin/activate

# Install the packages listed in requirements.txt file (located where yolov5 was installed)
pip install -r ~/PAI/detectors/yolov5/requirements.txt

# Deactivate virtual environment
deactivate
```

If you just need to update an existing environment:
```bash
module purge 
source ~/venv/yolov5/bin/activate
pip install -r ~/PAI/detectors/yolov5/requirements.txt # this will also update installed packages if applicable
deactivate
```

If you need to update YOLOv5, then just simply do a `git pull` in its repository:
```bash
cd ~/PAI/detectors/yolov5
git pull
```

# Data

## Download the raw data

Download the raw P1_dataset:
```bash
cd ~/datasets # move to home/datasets
wget https://portal.idiv.de/nextcloud/index.php/s/wN349pKaXn4ckg6/download # don’t add .zip
unzip download # unzips in a folder with its original name as stored on NextCloud: P1_Data
rm download # delete the zip file
```

## Prepare data

Will create a new directory named `P1_Data_sampled` in `~/datasets`. 
This folder respects the YOLOv5 data structure requirement.

First, make sure that the data preparation scripts respect the Linux path style. 
Replace any `\\` with a `/`, or use `os.path.join` (preffered).

```bash
source ~/venv/yolov5/bin/activate
cd ~/PAI/scripts/ # this must be the current directory when running utils_create_datasets.py
module load Python # case sensitive!
python ~/PAI/scripts/utils_create_datasets.py
deactivate
```

During the creation of the dataset, when executing `python ~/PAI/scripts/utils_create_datasets.py`, you should see something like:
```
Original dataset
/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/utils_datapaths.py:63: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  print_df = df.groupby(['class'])['images_path', 'labels_path'].count()
                        images_path  labels_path
class                                           
araneae                        1855         1523
coleoptera                     2490         2336
diptera                        2807         2401
hemiptera                      1991         1711
hymenoptera                    2994         2461
hymenoptera_formicidae         1474         1051
lepidoptera                    5102         4576
orthoptera                     1792         1649

Number of image tiles per class in 20.0% valdiation dataset
/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/utils_datasampling.py:46: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  print_df = df_test.groupby(['class'])['images_path', 'labels_path'].count()
                        images_path  labels_path
class                                           
araneae                         210          210
coleoptera                      210          210
diptera                         210          210
hemiptera                       210          210
hymenoptera                     210          210
hymenoptera_formicidae          210          210
lepidoptera                     210          210
orthoptera                      210          210

Number of image tiles per class in 20.0% valdiation dataset
/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/utils_datasampling.py:58: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  print_df = df_val.groupby(['class'])['images_path', 'labels_path'].count()
                        images_path  labels_path
class                                           
araneae                         210          210
coleoptera                      210          210
diptera                         210          210
hemiptera                       210          210
hymenoptera                     210          210
hymenoptera_formicidae          210          210
lepidoptera                     210          210
orthoptera                      210          210

Number of image tiles per class in training dataset
/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/utils_datasampling.py:66: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  print_df = df_train.groupby(['class'])['images_path', 'labels_path'].count()
                        images_path  labels_path
class                                           
araneae                        1103         1103
coleoptera                     1916         1916
diptera                        1981         1981
hemiptera                      1291         1291
hymenoptera                    2041         2041
hymenoptera_formicidae          631          631
lepidoptera                    4156         4156
orthoptera                     1229         1229
14348it [08:13, 29.08it/s] 
1680it [00:06, 268.87it/s]
1680it [00:06, 258.39it/s]
finished
```

FYI: On Linux, can also count the number of files in a folder:
```bash
ls ~/datasets/P1_Data_sampled/train/images | wc -l # 14348
ls ~/datasets/P1_Data_sampled/train/labels | wc -l # 14348

ls ~/datasets/P1_Data_sampled/test/images | wc -l  #  1680
ls ~/datasets/P1_Data_sampled/test/labels | wc -l  #  1680

ls ~/datasets/P1_Data_sampled/val/images | wc -l   #  1680
ls ~/datasets/P1_Data_sampled/val/labels | wc -l   #  1680
```

# Train a model

## YOLOv5

Create a folder that will contain the log files (*.log & *.err). We only need to create this folder once and then for each train can refer to its path in the SBATCH header of the job scrips:
```bash
cd ~/PAI/detectors
mkdir logs_train_jobs
```

Download the YOLOv5 pre-trained weights on the COCO dataset:
```bash
cd ~/PAI/detectors/yolov5
mkdir weights_v6_1
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n6.pt -P ~/PAI/detectors/yolov5/weights_v6_1/
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt -P ~/PAI/detectors/yolov5/weights_v6_1/
```

### Job scripts

A train job can be sent to the cluster using these scripts:

- `yolov5_train_n6.sh` with 'nano' yolov5n6.pt pretrained weights
- `yolov5_train_s6.sh` with 'small' yolov5s6.pt pretrained weights

We can send a train job to the cluster like this (make sure you have the right path and file name of the .sh script):
```bash
sbatch ~/PAI/scripts/cluster/yolov5_train_n6.sh
```

To see a job status: `squeue -u &#36;USER`

[comment]: # I had to use &#36; instead of dollar sign in the line example above because of this https://stackoverflow.com/a/71177841/5193830
[comment]: # For a markdown comment I followed this https://stackoverflow.com/a/32190021/5193830

To cancel a job: `scancel <jobid>`, e.g. `scancel 2216373`.

### SBATCH header options

Comments regarding the SBATCH header of a job script.

This is an example of an SBATCH header:
```bash
#!/bin/bash
#SBATCH --job-name=train_yolov5
#SBATCH --partition=clara-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:rtx2080ti:8
#SBATCH --mem-per-gpu=11G
#SBATCH --time=50:00:00
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/cluster/logs_train_jobs/%j.log
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/cluster/logs_train_jobs/%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END
```

`#!/bin/bash`: Tells the cluster that this is an executable job script.

`#SBATCH --job-name=train_yolov5`: Give a custom name for the job, like `train_yolov5`, that will be carried by the Slurm Workload Manager.

`#SBATCH --partition=clara-job`: Request for the Clara cluster.

`#SBATCH --nodes=1`: Number of requested nodes.

`#SBATCH --cpus-per-task=32`: Number of requested CPUs (32 is max per each Clara node). CPUs are important for data loading as "workers", so better have plenty, with max 8 per GPU as per this comment of Glenn Jocher (author of YOLOv5): "80% of CPUs assigned as workers, the remaining 20% free, with a maximum of 8 workers per GPU" - https://github.com/ultralytics/yolov5/issues/715#issuecomment-672563009

`#SBATCH --gres=gpu:rtx2080ti:8`: Type and number of GPUs. The GPU options on Clara are: a) NVIDIA GeForce RTX 2080 Ti, 11 Gb RAM (`gpu:rtx2080ti`); b) Nvidia Tesla V100 (`gpu:v100`), 32 Gb RAM.

`#SBATCH --mem-per-gpu=11G`: Request 11 Gb per GPU for the NVIDIA GeForce RTX 2080 Ti; or 32 Gb for Nvidia Tesla V100.

Note that, `--mem`, `--mem-per-cpu`, and `--mem-per-gpu` options are mutually exclusive. 
So, requesting `--mem-per-cpu=16G` will not work if you already requested `--mem-per-gpu=11G`.
The max memory per node is 16G per each of the 32 CPUs, so a total of 512 Gb/node. 
Might it be that having max of RAM/node is important for data caching?
Note that `--mem-per-cpu=16G` didn't work with the curent SBATCH header structure.

`#SBATCH --time=50:00:00`: Requested time, e.g. `50:00:00` = 50 hours. One needs to have an estimation of how much a cluster job can last. Better overestimate because the Slurm Workload Manager will automatically kill any job when it reaches its time limit.

`#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/cluster/logs_train_jobs/%j.log`: The path for storing the job-id.log file. Make sure this path exists beforing running a script because an incorrect or unexisting path will not trigger an error.

`#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/scripts/cluster/logs_train_jobs/%j.err`: Same as above, but for the job-id.err file.

`#SBATCH --mail-type=BEGIN,TIME_LIMIT,END`: Email options. Instruct the Slurm Workload Manager to send emails regarding when a job starts after being put in the waiting list (`BEGIN`), if it reached the time limit (`TIME_LIMIT`) and if it ended (`END`).

More details about the Slurm options can be found at https://slurm.schedmd.com/sbatch.html


### `train.py` options

Note that, all absolute paths in a job script (except in the SBATCH header) can also be written relative to the home directory with the tilde (~) symbol.
For example:
```
/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/yolov5 # can be written as
~/PAI/detectors/yolov5
```

In a job *.sh script, when calling `train.py`, we have these options:

`-m torch.distributed.run`: for parallel mode, multiple GPUs per node; -m stands for module-name.

`--nproc_per_node`: specifies how many GPUs to use.

`--weights`: path to predefined weights or custom weights. 
Make sure you have the most up to date ones. 
You can download them like: 
```
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt -P ~/PAI/detectors/yolov5/weights_v6_1/
```
You can check for the releases here: https://github.com/ultralytics/yolov5/releases Pick the needed version id and then use that in the `/download/v6.1` part of the download link in `wget`.

`--cfg:`, eg `--cfg yolov5m6.yaml` is the model configuration yaml file. It needs to match the `--weights` argument, so if you use an m6 model, make sure is m6 in both arguments. This might be needed only if you train from scratch and not using pre-trained COCO weights. See Start from Scratch section at https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results

`--data`: path to yaml file that contains the data paths

`--hyp`: yolov5 hyperparameter configuration yaml file from `yolov5/data/hyps/`. 
"In general the smaller models perform better with low augmentation and the larger models perform better with high augmentation." from: https://github.com/ultralytics/yolov5/issues/5236. 
"Nano and Small models use hyp.scratch-low.yaml hyps, all others use hyp.scratch-high.yaml." from: https://github.com/ultralytics/yolov5/releases - Pretrained Checkpoints - Table Notes (click to expand).
"Default hyperparameters are in hyp.scratch-low.yaml. We recommend you train with default hyperparameters first before thinking of modifying any." from: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results

`--epochs`: number of epochs;

`--batch-size`: when run in parallel is the total batch-size. It will be divided evenly to each GPU. Must be a multiple of the number of GPUs. For example, for 4 GPUs and a batch size of 16, then you need to give 4*16=64, like `--batch-size 64`;

`--imgsz`: image resolution, eg, 1280 means 1280 x 1280 pixels

`--cache`: cache images in ram (default) or disk; This might improve speed. But it run out of memory for `--cache ram`. See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#3-train 
See also some experiments at https://github.com/stark-t/PAI/issues/28

`--single-cls`: if there are multiple classes, then treat them all like one (https://github.com/ultralytics/yolov5/issues/128). We will not use this most probably anymore.

`--workers`: is max number of dataloader workers (per RANK in DDP mode). Per rank referes per GPU id (https://stackoverflow.com/a/58703819/5193830). How many CPUs per each requested GPU. Tip: "80% of cpus assigned as workers, the remaining 20% free, with a maximum of 8 workers per GPU" https://github.com/ultralytics/yolov5/issues/715#issuecomment-672563009

`--project` : Path to folder where to save the detection results. If not specified, then the default is `.../yolov5/runs/train`;

`--name`: name of the folder that will be created in the folder given in --project. If it happens to have multiple runs with the same name, YOLO can add an index to the end of name.