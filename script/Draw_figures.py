import sys
sys.path.append('/home/haiqwa/Documents/')

import os

from argparse import ArgumentParser
from KINGHQ.utils.utils import CSV,Figure

LOG_DIR = '/home/haiqwa/Documents/KINGHQ/log'

parser = ArgumentParser(description="I'm Figure Drawer owned by Haiquan Wang")

parser
parser.add_argument('-i',"--input", type=str, default= "",
                    help="The input CSV file name(without .csv)")
                    
parser.add_argument('-a',"--all",  action="store_true",
                    help="Handle all csv files in $KINGHQ/log/")
args = parser.parse_args()

def draw_from_csv(files):
    iter_acc = Figure('iteration-training accuracy','Iterations','Accuracy',
                    '/home/haiqwa/Documents/KINGHQ/figure/I-A.png')
    time_acc = Figure('time-training accuracy','Time','Accuracy',
                    '/home/haiqwa/Documents/KINGHQ/figure/T-A.png')
    for name, path in files.items():
        csv_file=CSV(path)
        iter_acc.add(csv_file('iterations'),csv_file('accuracy'),label=name)
        time_acc.add(csv_file('time'),csv_file('accuracy'),label=name)
    iter_acc.save()
    time_acc.save()


if args.all:
    # all files
    files={}
    for dirpath,_,filenames in os.walk(LOG_DIR):
        # dirpath: current root dir
        # dirnames: subdirs
        # filenames: all files in the current root dir
        for each in filenames:
            # eg. BSP: /home/haiqwa/Documents/KINGHQ/log/BSP.csv
            files[os.path.splitext(each)[0]]=os.path.join(dirpath,each)
    draw_from_csv(files)

else:
    # user decide
    if args.input:
        files={}
        filenames=args.input.split(',')
        for each in filenames:
            files[each] = os.path.join(LOG_DIR, each+'.csv')
        draw_from_csv(files)


    else:
        # no input
        print("="*20)
        print("Error: There's no input csv file")
        print("="*20)

