# !/bin/bash



# mpirun -np $1 \
#     -bind-to none -map-by slot \
#     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH\
#      -x HOROVOD_TIMELINE=~/Documents/pytorch_project/log/MNIST/timeline.json \
#      -x HOROVOD_TIMELINE_MARK_CYCLES=1\
#     -mca pml ob1 -mca btl ^openib \
#     python $2
python -m torch.distributed.launch \
    --nproc_per_node $1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 8889 \
    $2