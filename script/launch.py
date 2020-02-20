import os

ROOT_DIR = '/home/v-haiqwa/Documents/KINGHQ/'

from argparse import ArgumentParser
parser = ArgumentParser(description="I'm launcher script file of KINGHQ project owned by Haiquan Wang")
parser.add_argument('-s',"--server", type=str, default= "",
                    help="The server host configuration file")
                    
parser.add_argument('-w',"--worker",  type=str, default= "",
                    help="The worker host configuration file")

parser.add_argument('-m',"--master",  type=str, default= "",
                    help="The master host configuration file")

parser.add_argument('-c',"--consistency",  type=str, default= "",
                    help="The consistency model configuration file")

parser.add_argument('-i',"--input",  type=str, default= "",
                    help="The programme file")

args = parser.parse_args()


import subprocess

def load_file(path):
    result=[]
    with open(path,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        if line != '':
            result.append(line)
    return result

def excuteCommand(com):
    ex = subprocess.Popen(com, stdout=subprocess.PIPE, shell=True)
    out, err  = ex.communicate()
    status = ex.wait()
    return out.decode()

worker = load_file(ROOT_DIR + args.worker)
server = load_file(ROOT_DIR + args.server)
master = load_file(ROOT_DIR + args.master)

if not args.consistency:
    args.consistency = "BSP"

machine2role = dict()

for ip in worker:
    if ip in machine2role.keys():
        pass
    else:
        machine2role[ip] = []
    machine2role[ip].append("worker")
for ip in server:
    if ip in machine2role.keys():
        pass
    else:
        machine2role[ip] = []
    machine2role[ip].append("server")

if args.consistency == "ASP" or \
    args.consistency == "BSP":
    # no need for master
    pass
else:
    #SSP need for master
    ip = master[0]
    if ip in machine2role.keys():
        pass
    else:
        machine2role[ip] = []
    machine2role[ip].append("master")


# print(machine2role)

for ip,role_list in machine2role.items():
    for i in range(len(role_list)):
        role_list[i]+='\n'
    filepath = ROOT_DIR + "/config/send/" + ip 
    with open(filepath,'w') as f:
        f.writelines(role_list)
    excuteCommand('ssh v-haiqwa@'+ip+' '+'rm -rf /home/v-haiqwa/Documents/KINGHQ/config/recv')
    excuteCommand('ssh v-haiqwa@'+ip+' '+'mkdir /home/v-haiqwa/Documents/KINGHQ/config/recv')
    excuteCommand('scp '+filepath+' v-haiqwa@'+ip+':/home/v-haiqwa/Documents/KINGHQ/config/recv/')

    excuteCommand('ssh v-haiqwa@'+ip+' '+'rm -rf /home/v-haiqwa/Documents/KINGHQ/config/exefile')
    excuteCommand('ssh v-haiqwa@'+ip+' '+'mkdir /home/v-haiqwa/Documents/KINGHQ/config/exefile')
    excuteCommand('scp '+args.input+' v-haiqwa@'+ip+':/home/v-haiqwa/Documents/KINGHQ/config/recv/')
    


