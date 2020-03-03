# KINGHQ

It's a graduation project of Haiquan Wang.

**Background**
---
Nowadays, machine learning is getting more and more attention, including but not limited to
computer vision and natural language process. However, machine learning is a time-costing 
and resource-costing task for large model and dataset. Under this circumstances, distributed
machine learning becomes a necessary choice. Besides the raw distributed strategy inserted in
the popular frameworks, for example, pytorch, there are other optimization distributed strategy
also built on the current frameworks, such as horovod and byteps. However, they are mainly
designed for hard-synchronization(BSP), which means a worker has to wait for others to complete
communication in every iteration. Many computation resources are wasted during waiting, especailly
in the bad network situation. Therefore, some users may try to release the synchronization to achieve
speedup. For example, the worker will not be blocked if it doesn't exceed the slowest one over
specified steps(SSP). In addition, some researcher also want to release the synchronization to see
the performance of different models in large distributed environment.

Motivated by what is aforementioned, I design a flexible enough framework, in which users can
implement various synchronization strategies while they only need to modify little code. Besides,
based on this framework, I also implement many experiments to study the performance of DNN models
response to various synchronization strategies, like ASP and SSP. And I try to find some principles
that can inspire us what synchronization strategy is the best for specified DNN model.




**Getting Started**
Please build pytorch from the source.
>./run.sh
There is a simple launch command in ./run.sh. And usr can config the worker host and server host in ./config/host.
for example, in server hostfile:
>localhost
>localhost
And it will launch two servers in the localhost. 


