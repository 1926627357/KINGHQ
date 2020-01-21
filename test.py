# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd



# fig, ax = plt.subplots()

# csv_data=pd.read_csv('./log/ASP.csv')
# # print(csv_data)
# iterations=csv_data["iterations"].values
# accuracy=csv_data["accuracy"].values
# ax.plot(iterations, accuracy, label='ASP')

# csv_data=pd.read_csv('./log/BSP.csv')
# # print(csv_data)
# iterations=csv_data["iterations"].values
# accuracy=csv_data["accuracy"].values
# ax.plot(iterations, accuracy, label='BSP')

# ax.set(xlabel='iterations', ylabel='accuracy',
#        title='iterations-accuracy')
# ax.grid()
# ax.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=0, mode="expand",
#             borderaxespad=0., ncol=200, fontsize='small')
# fig.savefig("test0.png")

import random

number=[]

for _ in range(100):
    number.append(random.randint(1,10))

print(sorted(number))
