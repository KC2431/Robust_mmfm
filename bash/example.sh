#!/bin/bash
#SBATCH --job-name=Search
#SBATCH --chdir=/home/htc/mwagner/YOUR_DIRECTORY      # Navigate to the working directory where your script lies
#SBATCH --output=/home/htc/mwagner/SCRATCH/%j.log     # Standard output and error log
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu  # Specify the desired partition, e.g. gpu or big
#SBATCH --exclude=htc-gpu[020-023,037,038] # Only A40 GPU
#SBATCH --time=0-20:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --ntasks=1
#
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL@zib.com

echo 'Getting node information'
date;hostname;id;pwd

echo 'Activating virtual environment'
source ~/.bashrc
conda activate MA
which python

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Print GPUs'
/usr/bin/nvidia-smi

echo 'Running script'
python YOURSCRIPT.py
