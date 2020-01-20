import os
target=os.popen('lsof -i:8889').readlines()
pid_list=[]
for i in range(1,len(target)):
    pid_list.append(target[i].split(' ')[2])


for i in pid_list:
    os.system('kill '+i)
