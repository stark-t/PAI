#!/bin/bash

# Script to display information about the used environment and hardware.

# Display available memory.
# In the log files it looks like this indicates always the max available per node.
free -t -h
printf '\n' # print new line

# Disply NVIDIA System Management Interface and store it in the output log file.
nvidia-smi
printf '\n'

module purge

# Activate virtual environment
source ~/venv/yolov5/bin/activate

module load Python 

# Important for reproducibility session-info:
echo '========================================================================'
echo 'cat /etc/os-release'
echo '========================================================================'
cat /etc/os-release
printf '\n'

echo '========================================================================'
echo 'Linux Standard Base: lsb_release -a'
echo '========================================================================'
lsb_release -a
printf '\n'

echo '========================================================================'
echo 'Linux host: hostnamectl'
echo '========================================================================'
hostnamectl
printf '\n'

echo '========================================================================'
echo 'Information about the current locale'
echo '========================================================================'
locale
printf '\n'

echo '========================================================================'
echo 'List of Linux kernel modules currently loaded (in alphabetical order)'
echo '========================================================================'
lsmod | sort
printf '\n'

echo '========================================================================'
echo 'List of Python packages currently installed in the loaded environment (in alphabetical order)'
echo '========================================================================'
pip list # pip displays the packages and their verison in alphabetical order by default.
printf '\n'

# Deactivate virtual environment
deactivate