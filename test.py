# import torch
# import torchvision
# from torchvision import datasets
# import torchvision.transforms as transforms
# import torch.distributed as dist
# dist.init_process_group(backend='gloo')

# train_dataset = \
#     datasets.MNIST('~/Documents/pytorch_project/dataset/MNIST'+'data-%d'%dist.get_rank(), train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))

# train_sampler = torch.utils.data.distributed.DistributedSampler(
#     train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=16, sampler=train_sampler)

# train_sampler.set_epoch(2)
# train_loader=list(train_loader)

# print("rank: %d"%dist.get_rank()+str(train_loader[0][1]))

