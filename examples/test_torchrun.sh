#!/bin/bash

#SBATCH --job-name=normflow 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=kshdnormal
#SBATCH --nodes=4
#SBATCH --gres=dcu:4

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
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=0 


###nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
###nodes_array=($nodes)
###head_node=${nodes_array[0]}
###head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

####echo Node IP: $head_node_ip
####export LOGLEVEL=INFO
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo  $head_node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo head_node_ip
echo Node IP: $head_node_ip
NODE_RANK=$SLURM_NODEID
echo $NODE_RANK
export HIP_VISIBLE_DEVICES=0,1,2,3
# make sure nnodes and nproc_per_node are the same as slurm
srun torchrun \
--nnodes ${SLURM_NNODES} \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:45678 \
parquet_training.py \
--launcher=slurm
#parquet_training.py
#minimal_test2.py

#--rdzv_endpoint $head_node_ip:29500 \
