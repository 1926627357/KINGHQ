import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['1', '3', '5', '7']
KINGHQ = [21, 34, 30, 33]
Horovod = [21, 31, 48, 50]
Pytorch = [20, 46, 78, 90]
Linear_scaling = [21, 63, 106, 148]

x = np.arange(len(labels))  # the label locations
width = 0.17  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*3/2, KINGHQ, width, label='KINGHQ')
rects2 = ax.bar(x - width/2, Horovod, width, label='Horovod')
rects3 = ax.bar(x + width/2, Pytorch, width, label='Pytorch')
rects4 = ax.bar(x + width*3/2, Linear_scaling, width, label='Linear scaling')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Samples/second(x100)')
ax.set_xlabel('nodes')
ax.set_title('Scalability Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=0, mode="expand",
                    borderaxespad=0., ncol=100, fontsize='small')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

fig.savefig('/home/haiqwa/Documents/KINGHQ/figure/scale')