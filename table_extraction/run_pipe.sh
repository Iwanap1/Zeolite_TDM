#!/bin/bash
#PBS -lselect=1:ncpus=1:mem=32gb:ngpus=1
#PBS -lwalltime=3:30:0

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate zeolite-flash
cd $HOME/Zeolite_TDM/table_extraction
echo "Checking GPU availability..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"

python run.py

