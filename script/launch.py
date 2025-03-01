import os

ROOT_DIR = '/home/haiqwa/Documents/KINGHQ/'

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

master_worker=True
for ip in worker:
    if ip in machine2role.keys():
        pass
    else:
        machine2role[ip] = []
    if master_worker:
        machine2role[ip].append("masterworker")
        master_worker=False
    else:
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

print("*"*50)
print("PHASE 1 WORKLOAD ASSIGNMENT")
print("Begin to transform programme file and config for every machine!")


size=0
local_size=dict()
for ip,role_list in machine2role.items():
    size+=len(role_list)
    local_size[ip]=len(role_list)
    for i in range(len(role_list)):
        role_list[i]+='\n'
    filepath = ROOT_DIR + "/config/send/" + ip 
    with open(filepath,'w') as f:
        f.writelines(role_list)
    excuteCommand('ssh haiqwa@'+ip+' '+'rm -rf /home/haiqwa/Documents/KINGHQ/config/recv')
    excuteCommand('ssh haiqwa@'+ip+' '+'mkdir /home/haiqwa/Documents/KINGHQ/config/recv')
    excuteCommand('scp '+filepath+' haiqwa@'+ip+':/home/haiqwa/Documents/KINGHQ/config/recv/')

    excuteCommand('ssh haiqwa@'+ip+' '+'rm -rf /home/haiqwa/Documents/KINGHQ/config/exefile')
    excuteCommand('ssh haiqwa@'+ip+' '+'mkdir /home/haiqwa/Documents/KINGHQ/config/exefile')
    excuteCommand('scp '+args.input+' haiqwa@'+ip+':/home/haiqwa/Documents/KINGHQ/config/exefile/')

print("size: %d"%size)
print("local size ",local_size)
    

print("END")
print("*"*50)

MPI_COMMAND = "/home/haiqwa/anaconda3/envs/pytorch/bin/mpirun -np "+str(size)+" -H "

for ip,s in local_size.items():
    MPI_COMMAND+=ip+":"+str(s)+","
# delete the last str
MPI_COMMAND=MPI_COMMAND[:-1]

ip_prefix=ip.split('.')
ip_prefix[-1]='0'
ip_prefix='.'.join(ip_prefix)
ip_prefix=' -mca btl_tcp_if_include '+ip_prefix+'/24'


MPI_COMMAND+=ip_prefix+" /home/haiqwa/anaconda3/envs/pytorch/bin/python -B "+"/home/haiqwa/Documents/KINGHQ/config/exefile/"+args.input.split('/')[-1]

print(MPI_COMMAND)
# print(excuteCommand(MPI_COMMAND))
