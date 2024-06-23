#!/bin/bash

#SBATCH --job-name=normflow 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=titanv
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/tao_%j.out      # 标准输出文件



#export LD_LIBRARY_PATH=/public/home/taowang/miniconda3/envs/normflow/lib:$LD_LIBRARY_PATH
#export NCCL_IB_HCA=mlx5_0
#export NCCL_SOCKET_IFNAME=ib0
#export HSA_FORCE_FINE_GRAIN_PCIE=1
#export OMP_NUM_THREADS=1
#export NCCL_NET_GDR_LEVEL=0 


###nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
###nodes_array=($nodes)
###head_node=${nodes_array[0]}
###head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

####echo Node IP: $head_node_ip
####export LOGLEVEL=INFO
#conda init
#conda activate tao_normflfow
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo  $head_node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo head_node_ip
echo Node IP: $head_node_ip
NODE_RANK=$SLURM_NODEID
echo $NODE_RANK
#export HIP_VISIBLE_DEVICES=0,1,2,3
# make sure nnodes and nproc_per_node are the same as slurm
srun torchrun \
--nnodes ${SLURM_NNODES} \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
parquet_training.py \
--launcher=slurm
#--rdzv_endpoint $head_node_ip:60797 \
#parquet_training.py
#minimal_test2.py

#--rdzv_endpoint $head_node_ip:29500 \
