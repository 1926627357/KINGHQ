import os
import json5
###
#this script is used to update all machines project

path = "/home/v-haiqwa/Documents/KINGHQ/config/host/slaverlist"
with open(path,"r") as load_f:
    slaverlist=json5.load(load_f)

import subprocess

def excuteCommand(com):
    ex = subprocess.Popen(com, stdout=subprocess.PIPE, shell=True)
    out, err  = ex.communicate()
    status = ex.wait()
    return out.decode()

print(excuteCommand('cd /home/v-haiqwa/Documents/KINGHQ/ && git add . && git commit -m \"update\" && git push origin master'))
# os.popen('cd /home/v-haiqwa/Documents/KINGHQ/ && git add . && git commit -m \"update\" && git push origin master')
for hostname,ip in slaverlist.items():
    print("="*30)
    print(excuteCommand("ssh v-haiqwa@"+ip+" cd /home/v-haiqwa/Documents/KINGHQ/ && git pull"))