import json5
import json
from argparse import ArgumentParser

relative_path='/home/haiqwa/Documents/KINGHQ'

parser = ArgumentParser(description="I'm JSON process owned by Haiquan Wang")

parser.add_argument("--input", type=str, default= relative_path+"/usrJSON/demo.json",
                    help="The unprocessed JSON file path")
parser.add_argument("--output", type=str, default= relative_path+"/strategy/demo.json")


args = parser.parse_args()

with open(relative_path+"/strategy.json", "r") as load_f:
    # here, we read the strategy.json as a model
    model=json5.load(load_f)

with open(args.input, "r") as load_f:
    # here, strategy is a dict
    strategy=json5.load(load_f)





# network structure
if strategy['structure'] == 'Parameter Server':
    model['network']['structure']['parameter server'] = True
    model['network']['structure']['decentralized'] = False
elif strategy['structure'] == 'Decentralized':
    model['network']['structure']['parameter server'] = False
    model['network']['structure']['decentralized'] = True

if strategy['matrix']:
    model['network']['matrix'] = json5.loads(strategy['matrix'])
else:
    model['network']['matrix'] = []

# worker configuration
# barrier
model['worker']['barrier']['decision'] = strategy['barrier']

# worker pull
if strategy['pull_worker'] == 'Interval':
    model['worker']['pull']['when']['Interval']['decision'] = True
    model['worker']['pull']['when']['Interval']['interval'] = int(strategy['pull_worker_value_interval']) if strategy['pull_worker_value_interval'] else 0
    model['worker']['pull']['when']['staleness']['decision'] = False
elif strategy['pull_worker'] == 'Staleness':
    model['worker']['pull']['when']['Interval']['decision'] = False
    model['worker']['pull']['when']['staleness']['staleness'] = int(strategy['pull_worker_value_staleness']) if strategy['pull_worker_value_staleness'] else 0
    model['worker']['pull']['when']['staleness']['decision'] = True

if strategy['pull_what_worker'] == 'pull_what_worker1':
    model['worker']['pull']['what']['solu1']['decision']=True
    model['worker']['pull']['what']['solu2']['decision']=False
    model['worker']['pull']['what']['solu3']['decision']=False
    model['worker']['pull']['what']['solu4']['decision']=False
elif strategy['pull_what_worker'] == 'pull_what_worker2':
    model['worker']['pull']['what']['solu1']['decision']=False
    model['worker']['pull']['what']['solu2']['decision']=True
    model['worker']['pull']['what']['solu3']['decision']=False
    model['worker']['pull']['what']['solu4']['decision']=False
elif strategy['pull_what_worker'] == 'pull_what_worker3':
    model['worker']['pull']['what']['solu1']['decision']=False
    model['worker']['pull']['what']['solu2']['decision']=False
    model['worker']['pull']['what']['solu3']['decision']=True
    model['worker']['pull']['what']['solu4']['decision']=False
elif strategy['pull_what_worker'] == 'pull_what_worker4':
    model['worker']['pull']['what']['solu1']['decision']=False
    model['worker']['pull']['what']['solu2']['decision']=False
    model['worker']['pull']['what']['solu3']['decision']=False
    model['worker']['pull']['what']['solu4']['decision']=True

# worker push
model['worker']['push']['accumulate']['decision'] = strategy['Accum_push_worker']

model['worker']['push']['action']['Interval']['decision'] = strategy['Accum_push_interval_worker']
model['worker']['push']['action']['Interval']['interval'] = int(strategy['Accum_push_interval_value_worker']) if strategy['Accum_push_interval_value_worker'] else 0

if strategy['push_what_worker'] == 'push_what_worker1':
    model['worker']['push']['action']['what']['solu1']['decision']=True
    model['worker']['push']['action']['what']['solu2']['decision']=False
    model['worker']['push']['action']['what']['solu3']['decision']=False
    model['worker']['push']['action']['what']['solu4']['decision']=False
elif strategy['push_what_worker'] == 'push_what_worker2':
    model['worker']['push']['action']['what']['solu1']['decision']=False
    model['worker']['push']['action']['what']['solu2']['decision']=True
    model['worker']['push']['action']['what']['solu3']['decision']=False
    model['worker']['push']['action']['what']['solu4']['decision']=False
elif strategy['push_what_worker'] == 'push_what_worker3':
    model['worker']['push']['action']['what']['solu1']['decision']=False
    model['worker']['push']['action']['what']['solu2']['decision']=False
    model['worker']['push']['action']['what']['solu3']['decision']=True
    model['worker']['push']['action']['what']['solu4']['decision']=False
elif strategy['push_what_worker'] == 'push_what_worker4':
    model['worker']['push']['action']['what']['solu1']['decision']=False
    model['worker']['push']['action']['what']['solu2']['decision']=False
    model['worker']['push']['action']['what']['solu3']['decision']=False
    model['worker']['push']['action']['what']['solu4']['decision']=True

model['worker']['push']['action']['update']['decision'] = strategy['Update_clock']
model['worker']['push']['clear accumulate']['decision'] = strategy['Clear_Push_Accumulate']

# worker apply
model['worker']['apply']['accumulate']['decision'] = strategy['Accum_apply_worker']

model['worker']['apply']['action']['Interval']['decision'] = strategy['Accum_apply_interval_worker']
model['worker']['apply']['action']['Interval']['interval'] = int(strategy['Accum_apply_interval_value_worker']) if strategy['Accum_apply_interval_value_worker'] else 0


if strategy['apply_what_worker'] == 'apply_what_worker1':
    model['worker']['apply']['action']['what']['Accum_apply'] = True
    model['worker']['apply']['action']['what']['grads'] = False
elif strategy['apply_what_worker'] == 'apply_what_worker2':
    model['worker']['apply']['action']['what']['Accum_apply'] = False
    model['worker']['apply']['action']['what']['grads'] = True

model['worker']['apply']['action']['update']['decision'] = strategy['Update_version']
model['worker']['apply']['clear accumulate']['decision'] = strategy['Clear_Apply_Accumulate_worker']



# server configuration
# server check
if strategy['check'] == 'Staleness':
    model['server']['check']['staleness']['decision']=True

    model['server']['check']['staleness']['staleness']=int(strategy['check_staleness_value']) if strategy['check_staleness_value'] else 0
    model['server']['check']['version']['decision']=False
    model['server']['check']['default']['decision']=False
elif strategy['check'] == 'Version':
    model['server']['check']['staleness']['decision']=False
    model['server']['check']['version']['decision']=True
    model['server']['check']['default']['decision']=False
elif strategy['check'] == 'Default':
    model['server']['check']['staleness']['decision']=False
    model['server']['check']['version']['decision']=False
    model['server']['check']['default']['decision']=True

# server apply
model['server']['apply']['accumulate']['decision'] = strategy['Accum_push_server']
model['server']['apply']['staleness']['decision'] = strategy['Apply_Staleness']

model['server']['apply']['action']['Interval']['decision'] = strategy['Interval_server']

model['server']['apply']['action']['Interval']['interval'] = int(strategy['Interval_server_value']) if strategy['Interval_server_value'] else 0

if strategy['apply_what_server'] == 'apply_accum_server':
    model['server']['apply']['action']['what']['Accum_apply'] = True
    model['server']['apply']['action']['what']['grads'] = False
elif strategy['apply_what_server'] == 'apply_grads_server':
    model['server']['apply']['action']['what']['Accum_apply'] = False
    model['server']['apply']['action']['what']['grads'] = True


model['server']['apply']['action']['Interval']['average'] = strategy['averaged']


model['server']['apply']['action']['update']['decision'] = strategy['Update_global_version']
model['server']['apply']['clear accumulate']['decision'] = strategy['Clear_Apply_Accumulate']

# strategy change into str
strategy=json.dumps(model,indent=4)
with open(args.output,"w") as write_f:
    write_f.write(strategy)