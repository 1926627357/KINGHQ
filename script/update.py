import os
import json5


path = "/home/v-haiqwa/Documents/KINGHQ/config/host/slaverlist"
with open(path,"r") as load_f:
    slaverlist=json5.load(load_f)

os.popen("cd /home/v-haiqwa/Documents/KINGHQ/ && git add . && git commit -m \"update\" && git push origin master")
for hostname,ip in slaverlist.items():
    os.popen("ssh v-haiqwa@"+ip+" cd /home/v-haiqwa/Documents/KINGHQ/ && git pull")