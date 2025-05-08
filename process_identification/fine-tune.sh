#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=32gb:ngpus=1
#PBS -lwalltime=4:0:0

echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate phi3-qlora-flash

cd $HOME/Zeolite_TDM/process_identification
python fine-tune-phi.py

