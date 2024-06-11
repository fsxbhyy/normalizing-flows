#!/bin/sh
#SBATCH --job-name=normflow 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=kshdnormal
#SBATCH --nodes=2
#SBATCH --gres=dcu:2

source /public/home/taowang/miniconda3/etc/profile.d/conda.sh
module purge
module load compiler/devtoolset/7.3.1
module load    compiler/gcc/9.3.0  

module load mpi/hpcx/2.11.0/gcc-7.3.1
module load  compiler/dtk/23.10
# load the environment
#module unload compiler/dtk/22.10.1
#module load   compiler/dtk/22.10    
#clean the environment
# running the command
conda activate normflow 
export LD_LIBRARY_PATH=/public/home/taowang/miniconda3/envs/normflow/lib:$LD_LIBRARY_PATH

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=24

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

#srun python3 parquetAD.py
srun python3 parquet_training.py
