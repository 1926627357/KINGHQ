import os

pytorch_laucher0 = "/home/v-haiqwa/anaconda3/envs/pytorch/bin/python -m torch.distributed.launch \
    --nproc_per_node 2 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr 10.150.144.122 \
    --master_port 8889 \
    /home/v-haiqwa/Documents/KINGHQ/test/distributed.py"

pytorch_laucher1 = "/home/v-haiqwa/anaconda3/envs/pytorch/bin/python -m torch.distributed.launch \
    --nproc_per_node 2 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr 10.150.144.122 \
    --master_port 8889 \
    /home/v-haiqwa/Documents/KINGHQ/test/distributed.py"

# out_srgws04 = os.popen('ssh v-haiqwa@10.150.144.122 '+pytorch_laucher0)
out_srgws10 = os.popen('ssh v-haiqwa@10.190.175.223 '+pytorch_laucher1)

