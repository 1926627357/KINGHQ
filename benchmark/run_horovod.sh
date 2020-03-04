# !/bin/bash


# export HOROVOD_FUSION_THRESHOLD=0
# horovodrun -np 3 python simple_horovod.py
mpirun -np $1 \
    -H 10.0.0.51:1,10.0.0.49:1,10.0.0.50:1 -mca btl_tcp_if_include 10.0.0.0/24 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_FUSION_THRESHOLD=32\
    -mca pml ob1 -mca btl ^openib \
    python $2
# python -m torch.distributed.launch \
#     --nproc_per_node $1 \
#     --nnodes 1 \
#     --node_rank 0 \
#     --master_addr localhost \
#     --master_port 8889 \
#     $2