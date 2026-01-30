import faulthandler
faulthandler.enable()

import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import namedtuple
import dgl
import math
from tqdm import tqdm, trange
from time import time
import json
import sys
import openroad as ord
from openroad import Tech, Design, Timing
import copy
from pathlib import Path


from demo2_gate_sizing_helpers import *

import argparse
###############
#path argumant#
###############
# parser = argparse.ArgumentParser(description="path of your ASPDAC2024-Turotial clone (must include /ASPDAC2024-Turotial)")
# parser.add_argument("--path", type = Path, default='./', action = 'store')
# pyargs = parser.parse_args()
###################
#set up matplotlib#
###################
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display
##################################
#use gpu or cpu(cpu for tutorial)#
##################################
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#print(device)
########################################
#load design using openroad python apis#
########################################
# Load design into OpenROAD and initialize timing analysis
# Returns: ord_tech (Tech), ord_design (Design), timing (OpenSTA), db, chip, block, nets
# This function uses OpenROAD Python APIs to:
#   1. Load Liberty (.lib) and LEF files for technology information
#   2. Read Verilog netlist and link top module
#   3. Set clock constraints using TCL commands
#   4. Build cell_dict dynamically from the library (no JSON needed)
#   5. Return database objects for querying circuit properties
input_dir=sys.argv[1]
platform_dir=sys.argv[2]
output_dir=sys.argv[3]
top_module=sys.argv[4]
ord_tech, ord_design, timing, db, chip, block, nets, cell_dict, cell_name_dict = load_ISPD_design(input_dir, platform_dir, output_dir, top_module)

os.makedirs(os.path.dirname("dict_files/cell_dict.json"), exist_ok=True)
# Open the file in write mode ('w') and use json.dump()
with open("dict_files/cell_dict.json", "w") as json_file:
    json.dump(cell_dict, json_file, indent=4)
    
os.makedirs(os.path.dirname("dict_files/cell_name_dict.json"), exist_ok=True)
with open("dict_files/cell_name_dict.json", "w") as json_file:
    json.dump(cell_name_dict, json_file, indent=4)