#!/bin/bash
#SBATCH --job-name=yolov5_infer_cpu # give a custom name
#SBATCH --partition=clara-cpu # use clara-job for GPU usage and clara-cpu for cpu usage
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00 # e.g. 100:00:00 = 100 hours; 00:30:00 = 30 min
#SBATCH --output=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_inference_speed_jobs/%j.log # path for job-id.log file;
#SBATCH --error=/home/sc.uni-leipzig.de/sv127qyji/PAI/detectors/logs_inference_speed_jobs/%j.err # path for job-id.err file;
#SBATCH --mail-type=BEGIN,TIME_LIMIT,END # email options

# Start with a clean environment
module purge
# Load the needed modules from the software tree (same ones used when we created the environment)
module load Python/3.9.6-GCCcore-11.2.0
# Activate virtual environment
source ~/venv/yolov5/bin/activate

# Call the helper script session_info.sh which will print in the *.log file info 
# about the used environment and hardware.
source ~/PAI/scripts/cluster/session_info.sh yolov5
# The first and only argument here, passed to $1, is the environment name set at ~/venv/
# Use source instead of bash, so that session_info.sh describes the environment activated in this script 
# (the parent script from which is called). See https://askubuntu.com/a/965496/772524

cd ~/PAI/detectors/yolov5
python3 detect.py \
--weights ~/PAI/detectors/yolov5/runs/train/p1_w-n6_hyp-med_8b_300e/weights/best.pt \
--source ~/datasets/img_syrphidae_sample_2022_06_17 \
--imgsz 1280 \
--save-txt \
--save-conf \
--nosave \
--name infer_speed_syrphidae_220617_n6

# Deactivate virtual environment
deactivate

# Run in terminal with:
# sbatch ~/PAI/scripts/cluster/yolov5_n6_inference_speed_cpu.sh