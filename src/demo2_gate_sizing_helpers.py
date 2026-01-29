#BSD 3-Clause License
#
#Copyright (c) 2023, The Regents of the University of Minnesota
#
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import openroad as ord
import pdn, odb, utl
from openroad import Tech, Design, Timing
from collections import defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, RelGraphConv
import random
from itertools import count
import dgl
import copy
import os
from pathlib import Path
import sys
import glob
from glob import glob


# replay memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                        ('graph', 'action', 'next_state', 'reward'))
        
    # insert if not yet filled else treat it like a circular buffer and add.
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # random sampling for the training step
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

  def __init__(self, n_state, n_cells, n_features, device):
    super(DQN, self).__init__()
    self.conv1 = RelGraphConv(n_state,64,2)
    self.conv2 = RelGraphConv(64,64,2)
    self.conv3 = RelGraphConv(64,2,2)
    self.device = device
    self.n_state = n_state
    self.n_cells = n_cells
    self.n_features = n_features
  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def  forward(self, graph, x=None):
    if x is None:
      x = get_state(graph, self.n_state, self.n_cells, self.n_features).to(self.device)
    e = graph.edata['types']
    x = F.relu(self.conv1(graph, x, e))
    x = F.relu(self.conv2(graph, x, e))
    x = self.conv3(graph, x, e)
    # Get a list of actions that are not valid and ensure they cant be selected.
    mask = generate_masked_actions(graph)
    x = x*(~mask) + (mask)*(x.min() - 1)
    return x

def get_type(cell_type, cell_dict, cell_name_dict):
  """Extract cell index and size from full cell name.
  
  Handles ASAP7 naming: INVx2_ASAP7_75t_R -> base_name='INV', size='x2'
  """
  import re
  
  # Parse ASAP7 cell name: <CELLNAME>x<SIZE>_ASAP7_75t_<VT>
  match = re.match(r'([A-Za-z0-9_]+?)(x[p0-9]+(?:p[0-9]+)?)_ASAP7', cell_type)
  if not match:
    print(f"Could not parse cell type: {cell_type}")
    return None, None
  
  cell = match.group(1)  # Base name (e.g., 'INV')
  drive = match.group(2)  # Size suffix (e.g., 'x2')
  
  if cell in cell_name_dict:
    cell_values = cell_dict[cell_name_dict[cell]]
    if drive in cell_values['sizes']:
      idx = cell_values['sizes'].index(drive)
      return int(cell_name_dict[cell]), idx
    else:
      print("Drive strength "+drive+" not found in cell :"+cell)
      print("Possible sizes", cell_values['sizes'])
      return None,None
  else:
    print("cell: "+cell+" not in dictionary")
    return None,None

def pin_properties(dbpin, CLKset, ord_design, timing):
  """
  Extracts timing properties from OpenROAD for a given pin.
  
  How data is obtained from OpenROAD:
  1. Uses timing.getPinSlack() to get slack from OpenSTA timing engine
  2. Uses timing.getPinSlew() to get signal transition time
  3. Uses timing.getPortCap() to calculate total load capacitance
  
  Args:
    dbpin: OpenDB pin object (ITerm) to query
    CLKset: Clock period settings
    ord_design: OpenROAD Design object
    timing: OpenROAD Timing object (OpenSTA)
    
  Returns:
    tuple: (slack, slew, load) - timing properties for the pin
  """
  ITerms = dbpin.getNet().getITerms()
  
  # Get slack from OpenROAD timing engine (OpenSTA)
  # slack = required_time - arrival_time (negative means timing violation)
  # Query both rise and fall slack, take the worst (minimum)
  slack = min(timing.getPinSlack(dbpin, timing.Fall, timing.Max), timing.getPinSlack(dbpin, timing.Rise, timing.Max))
  if slack < -0.5*CLKset[0]:
    slack = 0
  
  # Get slew (transition time) from OpenROAD timing engine
  slew = timing.getPinSlew(dbpin)  
  
  # Calculate total load capacitance driven by this pin
  # Sum up input capacitances of all fanout pins across all timing corners
  load = 0
  for ITerm in ITerms:
    if ITerm.isInputSignal():
      new_load = 0
      # Check all timing corners (e.g., fast, typical, slow) and use max load
      for corner in timing.getCorners():
        tmp_load = timing.getPortCap(ITerm, corner, timing.Max)
        if tmp_load > new_load:
          new_load = tmp_load
      load += new_load

  return slack, slew, load

def min_slack(dbpin, timing):
  slack = min(timing.getPinSlack(dbpin, timing.Fall, timing.Max), timing.getPinSlack(dbpin, timing.Rise, timing.Max))
  return slack

def generate_masked_actions(graph):
  # max size keep track of the index of the maximum size.
  # If the current size is maximum size we mask it out as an action
  upper_mask = graph.ndata['cell_types'][:,1] >= graph.ndata['max_size']-1
  lower_mask = graph.ndata['cell_types'][:,1] == 0
  # if the criteria for the mask is met we replace it with the minimum
  # to make sure that that action is never chosen
  mask = torch.cat((upper_mask.view(-1,1), lower_mask.view(-1,1)),1)
  return mask

def update_lambda(initial_lambda, slacks, K):
    Slack_Lambda = initial_lambda * ((1-slacks)**K)
    return Slack_Lambda

def optimize_model(memory, BATCH_SIZE, device, GAMMA, policy_net,\
                  target_net, optimizer, loss_history):
  if len(memory) < BATCH_SIZE:
    return optimizer, loss_history
  transitions = memory.sample(BATCH_SIZE)
  Transition = namedtuple('Transition', ('graph', 'action', 'next_state', 'reward'))
  batch = Transition(*zip(*transitions))
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = torch.zeros(BATCH_SIZE, device=device,dtype=torch.float32)
  for n_state, graph in enumerate(batch.graph):
    actions = policy_net(graph)
    state_action_values[n_state] = actions.view(-1)[action_batch[n_state,0]]
  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(BATCH_SIZE, device=device,dtype=torch.float32)

  for n_state, state in enumerate(batch.next_state):
    if state is not None:
      graph = batch.graph[n_state]
      state_g = state.to(device)
      next_state_values[n_state] = target_net(graph, state_g.view(graph.num_nodes(),-1)).max().detach()
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch

  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  loss_history.append(loss.item())
  optimizer.step()
  return optimizer, loss_history

def select_action(graph, inference = False, total_taken = False,\
                  steps_done = False, random_taken = False, policy_net = False,\
                  EPS_END = False, EPS_START = False, EPS_DECAY = False, device = False):
  total_taken +=1
  if inference:
    with torch.no_grad():
      action = policy_net(graph)
      return torch.argmax(action.view(-1)).view(1,1), total_taken, steps_done, random_taken

  sample = random.random()
  eps_threshold = EPS_END + (EPS_START - EPS_END) * 0.95**(steps_done / EPS_DECAY)
  steps_done += 1
  #get the mask
  mask = generate_masked_actions(graph)

  if int(sum(~mask.view(-1)))==0 :
    return -1, total_taken, steps_done, random_taken
  #Threshold keeps decreasing, so over time it takes more from the policy net.
  if sample > eps_threshold:
    with torch.no_grad():
      action = policy_net(graph)
      return torch.argmax(action.view(-1)).view(1,1), total_taken, steps_done, random_taken
  else:
    action = torch.randn_like(mask,dtype=torch.float32)
    action = (action-action.min()+1)*(~mask)
    random_taken+=1
    return torch.tensor([[torch.argmax(action.view(-1))]], device=device, dtype=torch.long),\
            total_taken, steps_done, random_taken

def get_subgraph(graph, nodes):
  node_set = {x.item() for x in nodes}
  #level 1
  in_nodes, _ = graph.in_edges(list(node_set))
  _, out_nodes = graph.out_edges(list(node_set))
  node_set.update(in_nodes.tolist())
  node_set.update(out_nodes.tolist())
  #level 2
  in_nodes, _ = graph.in_edges(list(node_set))
  _, out_nodes = graph.out_edges(list(node_set))
  node_set.update(in_nodes.tolist())
  node_set.update(out_nodes.tolist())

  subgraph = dgl.node_subgraph(graph, list(node_set))

  return subgraph

def get_critical_path_nodes(graph, ep_num, TOP_N_NODES, n_cells):
  """
  Identifies critical path nodes (gates with worst timing slack).
  
  How critical paths are found:
  1. Dynamically calculates the number of nodes to consider based on episode number
  2. Uses torch.topk() to find nodes with the smallest (most negative) slack values
  3. Filters to keep only nodes with negative slack (timing violations)
  4. If no violations exist, returns all nodes for area optimization
  
  Args:
    graph: DGL graph containing circuit netlist with slack values in ndata
    ep_num: Current episode number (increases scope over time)
    TOP_N_NODES: Base number of critical nodes (default: 100)
    n_cells: Number of cell types
    
  Returns:
    critical_path: Tensor of node indices representing critical path nodes
  """
  # Gradually increase the number of critical nodes considered as training progresses
  topk = min(len(graph.ndata['slack'])-1 , int(TOP_N_NODES*(1+0.01*ep_num)))
  
  # Find nodes with smallest slack values using torch.topk (largest=False gets minimums)
  # slack values come from OpenROAD timing analysis via getPinSlack() API
  min_slacks, critical_path = torch.topk(graph.ndata['slack'], topk, largest=False)
  
  # Keep only nodes with negative slack (timing violations)
  critical_path = critical_path[min_slacks<0]

  # Fallback: if no timing violations, optimize all nodes for area
  if critical_path.numel() <=0:
    critical_path = torch.arange(0,graph.num_nodes())

  return critical_path

def get_state(graph, n_state, n_cells, n_features):
  state = torch.zeros(graph.num_nodes(), n_state)
  state[:,-1] = graph.ndata['area']
  state[:,-2] = graph.ndata['slack']
  state[:,-3] = graph.ndata['slew']
  state[:,-4] = graph.ndata['load']
  state[:,:-n_features] =F.one_hot(graph.ndata['cell_types'][:,0],n_cells)*graph.ndata['cell_types'][:,1:2]
  return state

def env_step(episode_G, graph, state, action, CLKset, ord_design, timing,\
            cell_dict, norm_data, inst_names, episode_inst_dict, inst_dict,\
            n_cells, n_features, block, device, Slack_Lambda, eps):
  next_state = state.clone()
  reward = 0
  done =0
  #based on the selected action you choose the approriate cell and upsize it or downsize
  cell_sub = int(action/2)
  cell = graph.ndata['_ID'][cell_sub].item()
  inst_name = inst_names[cell]
  cell_size = episode_inst_dict[inst_name]['cell_type'][1]
  cell_idx = episode_inst_dict[inst_name]['cell_type'][0]
  dbpin = block.findITerm(inst_name + cell_dict[str(inst_dict[inst_name]['cell_type'][0])]['out_pin'])
  old_slack = min_slack(dbpin, timing)
  o_master_name = cell_dict[str(cell_idx)]['name']+\
                  cell_dict[str(cell_idx)]['sizes'][cell_size]
  if(action%2 == 0):
      cell_size +=1
  else:
      cell_size -=1
  if(cell_size>=cell_dict[str(cell_idx)]['n_sizes']):
    print("Above max")
    print(action,cell_dict[str(cell_idx)]['n_sizes'], cell_idx, cell_size)
  if(cell_size<0):
    print("below min")
    print(action,cell_dict[str(cell_idx)]['n_sizes'], cell_idx, cell_size)
  episode_inst_dict[inst_name]['cell_type'] = (cell_idx,cell_size)
  size = cell_dict[str(cell_idx)]['sizesi'][cell_size] #actual size

  #one hot encode the relavant feature with the magnitude of size.
  next_state[cell_sub,:-n_features] = F.one_hot(torch.tensor([cell_idx]),n_cells)*size
  episode_G.ndata['cell_types'][cell] = torch.tensor((cell_idx,cell_size))

  #replace the master node in the code and find the new slack,
  inst = block.findInst(inst_name)
  n_master_name = cell_dict[str(cell_idx)]['name']+\
                  cell_dict[str(cell_idx)]['sizes'][cell_size]
  db = ord.get_db()
  n_master = db.findMaster(n_master_name)
  inst.swapMaster(n_master)
  dbpin = block.findITerm(inst_name + cell_dict[str(inst_dict[inst_name]['cell_type'][0])]['out_pin'])
  new_slack = min_slack(dbpin, timing)

  old_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
  episode_inst_dict[inst_name]['area']= n_master.getWidth() * n_master.getHeight()
  new_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
  episode_G.ndata['area'][cell] = new_area

  # update_area
  next_state[cell_sub,-1] = new_area
  reward += torch.tensor(old_area-new_area)
  old_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slews = torch.zeros(len(episode_inst_dict.keys()))
  new_loads = torch.zeros(len(episode_inst_dict.keys()))

  for n, inst in inst_names.items():
    old_slacks[n] = episode_inst_dict[inst]['slack']
    tmp_db_pin = block.findITerm(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
      episode_inst_dict[inst]['slew'],
      episode_inst_dict[inst]['load']) = pin_properties(tmp_db_pin, CLKset, ord_design, timing)
      new_slacks[n] = episode_inst_dict[inst]['slack']
      new_slews[n] = episode_inst_dict[inst]['slew']
      new_loads[n] = episode_inst_dict[inst]['load']
  episode_G.ndata['slack'] = new_slacks.to(device)/norm_data['clk_period']
  for i in range(len(episode_G.ndata['slack'])):
    if episode_G.ndata['slack'][i] > 1:
      episode_G.ndata['slack'][i] = 1
  episode_G.ndata['slack'][torch.isinf(episode_G.ndata['slack'])] = 1
  episode_G.ndata['slew'] = new_slews.to(device)/ norm_data['max_slew']
  episode_G.ndata['load'] = new_loads.to(device)/ norm_data['max_load']

  next_state[:,-2] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['slack']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['clk_period']
  next_state[:,-3] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['slew']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['max_slew']
  next_state[:,-4] = torch.tensor([episode_inst_dict[inst_names[x.item()]]['load']
                                   for x in graph.ndata['_ID']])\
                                  / norm_data['max_load']
  next_state[torch.isinf(next_state[:,-2]),-2] = 1 # remove infinity from primary outputs. hadle it better.

  #Check TNS
  new_TNS = torch.min(new_slacks,torch.zeros_like(new_slacks))
  new_TNS = Slack_Lambda*new_TNS

  old_TNS = torch.min(old_slacks,torch.zeros_like(old_slacks))
  old_TNS = Slack_Lambda*old_TNS

  factor = torch.max(torch.abs(0.1*old_TNS), eps*torch.ones_like(old_TNS))
  factor = torch.max(torch.ones_like(old_TNS), 1/factor)
  reward += (torch.sum((new_TNS - old_TNS) * factor)).item()

  return reward, done, next_state, episode_inst_dict, episode_G

def env_reset(reset_state = None, episode_num = None, cell_name_dict = None,\
              CLKset = None, ord_design = None, timing = None, G = None,\
              inst_dict = None, CLK_DECAY = None, CLK_DECAY_STRT = None,\
              clk_init = None, clk_range = None, clk_final = None, inst_names = None,\
              block = None, cell_dict = None, norm_data = None, device = None):
  episode_G = copy.deepcopy(G)
  episode_inst_dict = copy.deepcopy(inst_dict)

  # if episode_num is not None:

  #   if episode_num<CLK_DECAY_STRT:
  #     clk = clk_init
  #   elif episode_num<CLK_DECAY+CLK_DECAY_STRT:
  #     clk = clk_init - clk_range*(episode_num -CLK_DECAY_STRT) /CLK_DECAY
  #   else:
  #     clk = clk_final
  #   ord_design.evalTclString("create_clock [get_ports i_clk] -name core_clock -period " + str(clk*1e-9))

  for i in range(len(inst_names)):
    inst_name = inst_names[i]
    inst = block.findInst(inst_name)
    if reset_state is not None:
      o_master_name = reset_state[i]
      cell_idx, cell_size = get_type(o_master_name, cell_dict, cell_name_dict)
      episode_inst_dict[inst_name]['cell_type'] = (cell_idx,cell_size)
      episode_G.ndata['cell_types'][i] = torch.tensor((cell_idx,cell_size))
    else:
      cell_size = episode_G.ndata['cell_types'][i,1].item()
      cell_idx = episode_G.ndata['cell_types'][i,0].item()
      o_master_name = cell_dict[str(cell_idx)]['name']+\
              cell_dict[str(cell_idx)]['sizes'][cell_size]

    db = ord.get_db()
    o_master = db.findMaster(o_master_name)
    if o_master_name != inst.getMaster().getName():
      inst.swapMaster(o_master)
    if reset_state is not None:
      episode_inst_dict[inst_name]['area']= o_master.getWidth() * o_master.getHeight()
      new_area = episode_inst_dict[inst_name]['area']/ norm_data['max_area']
      episode_G.ndata['area'][i] = new_area

  #if reset_state is not None:
  new_slacks = torch.zeros(len(episode_inst_dict.keys()))
  new_slews = torch.zeros(len(episode_inst_dict.keys()))
  new_loads = torch.zeros(len(episode_inst_dict.keys()))

  for n, inst in inst_names.items():
    tmp_db_pin = block.findITerm(inst+cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
      episode_inst_dict[inst]['slew'],
      episode_inst_dict[inst]['load']) = pin_properties(tmp_db_pin, CLKset, ord_design, timing)
      new_slacks[n] = episode_inst_dict[inst]['slack']
      new_slews[n] = episode_inst_dict[inst]['slew']
      new_loads[n] = episode_inst_dict[inst]['load']
  episode_G.ndata['slack'] = new_slacks.to(device)/norm_data['clk_period']
  for i in range(len(episode_G.ndata['slack'])):
    if episode_G.ndata['slack'][i] > 1:
      episode_G.ndata['slack'][i] = 1
  episode_G.ndata['slack'][torch.isinf(episode_G.ndata['slack'])] = 1
  episode_G.ndata['slew'] = new_slews.to(device)/ norm_data['max_slew']
  episode_G.ndata['load'] = new_loads.to(device)/ norm_data['max_load']

  return episode_G, episode_inst_dict, clk


def calc_cost(ep_G, Slack_Lambda):
  cost = torch.sum(ep_G.ndata['area'].to('cpu'))
  x = ep_G.ndata['slack'].to('cpu')
  new_slacks = torch.min(x, torch.zeros_like(x))
  cost += torch.sum(Slack_Lambda*(-new_slacks))
  return cost

def get_state_cells(ep_dict, inst_names, cell_dict):
  cells = []
  for x in inst_names.values():
    cell_size = ep_dict[x]['cell_type'][1]
    cell_idx = ep_dict[x]['cell_type'][0]
    cell_name = cell_dict[str(cell_idx)]['name']+\
                cell_dict[str(cell_idx)]['sizes'][cell_size]
    cells.append(cell_name)
  return cells

def pareto(pareto_points, pareto_cells, area, clk, ep_dict, inst_names,\
            cell_dict, inst_dict, block, ord_design, timing):
  cells = get_state_cells(ep_dict, inst_names, cell_dict)
  if len(pareto_points) <= 0:
    pareto_points.append((area, clk))
    pareto_cells.append(cells)
    return 1
  dominated_points = set()
  for n, pareto_point in enumerate(pareto_points):
    # if new point is dominated we skip
    if pareto_point[0] <= area and  pareto_point[1] <= clk:
      return 0
    # if new point dominates any other point
    elif pareto_point[0] >= area and  pareto_point[1] >= clk:
      dominated_points.add(n)

  pareto_points.append((area, clk))
  pareto_cells.append(cells)
  pareto_points = [val for n, val in enumerate(pareto_points) if n not in dominated_points]
  pareto_cells = [val for n, val in enumerate(pareto_cells) if n not in dominated_points]
  slacks = [min_slack(block.findITerm(x + cell_dict[str(inst_dict[x]['cell_type'][0])]['out_pin']), timing) for x in inst_names.values()]
  test_sl = np.min(slacks)
  return 1

def rmdir(directory):
  directory=Path(directory)
  for  item in directory.iterdir():
    if item.is_dir():
      rmdir(directory)
    else:
      item.unlink()
  directory.rmdir()

unit_micron = 2000
design = 'pid'
semi_opt_clk = '0.65'
clock_name = "i_clk"
CLK_DECAY=100; CLK_DECAY_STRT=25
n_features = 4
BATCH_SIZE = 64#128
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 200
LR = 0.001
BUF_SIZE = 1500#10000
STOP_BADS = 50
MAX_STEPS = 50#150#200 c432#51#300
TARGET_UPDATE = MAX_STEPS*25 #MAX_STEPS*5
EPISODE = 2 #150 # c880 #50 c432 #15#200
LOADTH = 0
DELTA = 0.000001
UPDATE_STOP = 250

TOP_N_NODES = 100
eps = 1e-5
inference = False
retrain = False
max_TNS = -100
max_WNS = -100
count_bads = 0
best_delay = 5
min_working_clk = 100
K = 4# set seprately to critical and non critical # accelration factor
CLKset = [0.6]
clk_final = CLKset[0]
clk_range = 0.98*(float(semi_opt_clk) - CLKset[0])
clk_init = clk_final + clk_range

def load_ISPD_design(input_dir, platform_dir, output_dir, top_module):
  """
  Loads the design into OpenROAD and initializes timing analysis.
  
  How data is obtained from OpenROAD:
  1. Creates Tech object and loads Liberty (.lib) and LEF files for technology info
  2. Creates Design object and loads Verilog netlist
  3. Creates Timing object (OpenSTA) for timing analysis
  4. Sets clock constraints using TCL commands
  5. Builds cell_dict dynamically from library (no JSON needed)
  6. Returns database objects for querying circuit properties
  
  Args:
    input_dir: Path to input files (verilog, def, sdc)
    platform_dir: Path to platform files (lib, lef, util)
    output_dir: Path for output files
    top_module: Top module name
    
  Returns:
    tuple: (ord_tech, ord_design, timing, db, chip, block, nets, cell_dict, cell_name_dict)
           - ord_tech: Technology information
           - ord_design: Design object
           - timing: Timing analysis engine (OpenSTA)
           - db: OpenDB database
           - chip: Top-level chip object
           - block: Design block with instances and nets
           - nets: List of all nets in the design
           - cell_dict: Cell library dictionary (built from OpenROAD)
           - cell_name_dict: Cell name lookup table
  """
  # Create technology object
  ord_tech = Tech()
  
  # Define paths to technology files
  lef_dir = Path(platform_dir) / "lef"
  lib_dir = Path(platform_dir) / "lib"
  util_dir = Path(platform_dir) / "util"

  lefFiles = lef_dir.glob("*.lef")
  libFiles = lib_dir.glob("*.lib")
  # tech_lef_file = path/"platforms/lef/NangateOpenCellLibrary.tech.lef"
  
  # Load technology files into OpenROAD
  # (The techlef file is included in Platform/ASAP7/lef along with the designs' lef files, so we read all the lef files in the lef directory)
  for lef in lefFiles:
      print("Lef path: ", str(lef))
      ord_tech.readLef(str(lef))
      print("Read lef: ", str(lef))

  # read liberty files    
  for lib in libFiles:
      print("Lib path: ", str(lib))
      ord_tech.readLiberty(str(lib))
      print("Read lib: ", str(lib))
  
  # Create design and timing analysis objects
  ord_design = Design(ord_tech)
  
  # Load netlist (Verilog file)
  # ord_design.readVerilog(f"{input_dir}/contest.v")
  # ord_design.link(top_module)  # Link top module
  
  ord_design.evalTclString(f"read_verilog {input_dir}/contest.v")
  print("Read verilog")
  ord_design.evalTclString(f"read_def {input_dir}/contest.def")
  print("Read def")
  
  # read sdc file
  sdcFile = f"{input_dir}/contest.sdc"
  ord_design.evalTclString("read_sdc %s"%sdcFile)
  print("Read SDC")
  
  # read rc file
  rcFile = f"{util_dir}/setRC.tcl"
  # ord_design.evalTclString("set rc_file %s"%rcFile)
  ord_design.evalTclString(f"source {rcFile}")
  print("Read rc")

  # set units
  ord_design.evalTclString("set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um -power mW")
  ord_design.evalTclString("set_units -power mW")
  
  # Get OpenDB database objects for querying and modification
  db = ord.get_db()
  chip = db.getChip()
  block = ord.get_db_block()  # Contains instances, nets, pins
  nets = block.getNets()  # Get all nets in the design
  
  timing = Timing(ord_design)  # OpenSTA timing engine
  
  # Build cell dictionary dynamically from OpenROAD library (no JSON needed)
  cell_dict, cell_name_dict = build_cell_dict_from_openroad(db)
  print(f"Built cell dictionary with {len(cell_dict)} cell types")
  
  return ord_tech, ord_design, timing, db, chip, block, nets, cell_dict, cell_name_dict

def load_design(path):
  """
  Loads the design into OpenROAD and initializes timing analysis.
  
  How data is obtained from OpenROAD:
  1. Creates Tech object and loads Liberty (.lib) and LEF files for technology info
  2. Creates Design object and loads Verilog netlist
  3. Creates Timing object (OpenSTA) for timing analysis
  4. Sets clock constraints using TCL commands
  5. Builds cell_dict dynamically from library (no JSON needed)
  6. Returns database objects for querying circuit properties
  
  Args:
    path: Path to the tutorial directory
    
  Returns:
    tuple: (ord_tech, ord_design, timing, db, chip, block, nets, cell_dict, cell_name_dict)
           - ord_tech: Technology information
           - ord_design: Design object
           - timing: Timing analysis engine (OpenSTA)
           - db: OpenDB database
           - chip: Top-level chip object
           - block: Design block with instances and nets
           - nets: List of all nets in the design
           - cell_dict: Cell library dictionary (built from OpenROAD)
           - cell_name_dict: Cell name lookup table
  """
  # Create technology object
  ord_tech = Tech()
  
  # Define paths to technology files
  lib_file = path/"platforms/lib/NangateOpenCellLibrary_typical.lib"
  lef_file = path/"platforms/lef/NangateOpenCellLibrary.macro.lef"
  tech_lef_file = path/"platforms/lef/NangateOpenCellLibrary.tech.lef"
  
  # Load technology files into OpenROAD
  # Liberty file: timing and power characterization
  ord_tech.readLiberty(lib_file.as_posix())
  # LEF files: physical layout information (layers, cells)
  ord_tech.readLef(tech_lef_file.as_posix())
  ord_tech.readLef(lef_file.as_posix())
  
  # Create design and timing analysis objects
  ord_design = Design(ord_tech)
  timing = Timing(ord_design)  # OpenSTA timing engine
  
  # Load netlist (Verilog file)
  design_file = path/f"designs/{design}_{semi_opt_clk}.v"
  ord_design.readVerilog(design_file.as_posix())
  ord_design.link(design)  # Link top module
  
  # Set clock constraint using TCL command through OpenROAD
  ord_design.evalTclString("create_clock [get_ports i_clk] -name core_clock -period " + str(clk_init*1e-9))
  
  # Get OpenDB database objects for querying and modification
  db = ord.get_db()
  chip = db.getChip()
  block = ord.get_db_block()  # Contains instances, nets, pins
  nets = block.getNets()  # Get all nets in the design
  
  # Build cell dictionary dynamically from OpenROAD library (no JSON needed)
  cell_dict, cell_name_dict = build_cell_dict_from_openroad(db)
  print(f"Built cell dictionary with {len(cell_dict)} cell types")
  
  return ord_tech, ord_design, timing, db, chip, block, nets, cell_dict, cell_name_dict


def build_cell_dict_from_openroad(db):
  """
  Builds cell dictionary dynamically from OpenROAD's database, eliminating the need for JSON parsing.
  
  How data is obtained from OpenROAD:
  1. Uses db.getLibs() to get all libraries, then lib.getMasters() to get all library cells
  2. For each master cell:
     - Groups cells by base name (e.g., "INVx2_ASAP7_75t_R", "INVx4_ASAP7_75t_R" -> "INV")
     - Extracts size suffix (e.g., "x2", "x4", "xp33" for 0.33x)
     - Gets output pin name using master.getMTerms()
  3. Organizes cells by type index and size variants
  
  Args:
    db: OpenDB database object
    
  Returns:
    tuple: (cell_dict, cell_name_dict)
           - cell_dict: Dictionary indexed by cell type index containing:
               * name: Base cell name (e.g., "INV")
               * sizes: List of size suffixes (e.g., ["xp33", "x1", "x2", "x4"])
               * sizesi: List of size floats (e.g., [0.33, 1, 2, 4])
               * n_sizes: Number of size variants
               * out_pin: Output pin name (e.g., "/Y")
               * c_in: List of placeholder zeros (actual capacitance extracted during timing analysis)
           - cell_name_dict: Lookup table mapping cell base name to index
  """
  from collections import defaultdict
  import re
  
  # Get all library cells from OpenROAD database
  # Access masters through libraries
  libs = db.getLibs()
  masters = []
  for lib in libs:
    masters.extend(lib.getMasters())
  
  # Group cells by base name (without size suffix)
  cell_groups = defaultdict(list)
  
  for master in masters:
    master_name = master.getName()
    # Skip special cells (filler, decap, tap, antenna - but keep TIEHI/TIELO)
    if any(x in master_name.upper() for x in ["FILL", "DECAP", "ANTENNA", "TAP"]):
      continue
    
    # Extract base name and size for ASAP7 cells (e.g., "INVx2_ASAP7_75t_R" -> "INV", "x2")
    # Pattern: <CELLNAME>x<SIZE>_ASAP7_75t_<VT>
    match = re.match(r'([A-Za-z0-9_]+?)(x[p0-9]+(?:p[0-9]+)?)_ASAP7', master_name)
    if match:
      base_name = match.group(1)
      size_suffix = match.group(2)
      cell_groups[base_name].append((size_suffix, master))
  
  # Build cell_dict with indexed entries
  cell_dict = {}
  cell_name_dict = {}
  
  def parse_size(size_suffix):
    """Convert size suffix to float (e.g., 'x2' -> 2.0, 'xp33' -> 0.33, 'x1p5' -> 1.5)"""
    size_str = size_suffix[1:]  # Remove 'x' prefix
    if 'p' in size_str:
      # Handle decimal notation (xp33 -> 0.33, x1p5 -> 1.5)
      parts = size_str.split('p')
      if parts[0]:  # x1p5 case
        return float(parts[0]) + float(parts[1]) / (10 ** len(parts[1]))
      else:  # xp33 case
        return float(parts[1]) / (10 ** len(parts[1]))
    else:
      return float(size_str)
  
  for idx, (base_name, cells) in enumerate(sorted(cell_groups.items())):
    # Sort cells by size
    cells_sorted = sorted(cells, key=lambda x: parse_size(x[0]))
    
    sizes = []
    sizesi = []
    c_in_list = []
    out_pin = None
    
    for size_suffix, master in cells_sorted:
      sizes.append(size_suffix)
      # Extract numeric size
      size_num = parse_size(size_suffix)
      sizesi.append(size_num)
      
      # Get output pin name
      if out_pin is None:
        for mterm in master.getMTerms():
          io_type = mterm.getIoType()
          # Handle both string and enum types
          io_type_str = io_type if isinstance(io_type, str) else str(io_type)
          if "OUTPUT" in io_type_str:
            out_pin = "/" + mterm.getName()
            break
      
      # Note: Input capacitance values (c_in) are not extracted here as they require
      # timing library information that's better obtained during actual timing analysis.
      # Using placeholder 0.0 - the actual values aren't critical for gate sizing algorithm
      c_in_list.append(0.0)
    
    # Store in cell_dict
    cell_dict[str(idx)] = {
      "name": base_name,
      "sizes": sizes,
      "sizesi": sizesi,
      "n_sizes": len(sizes),
      "out_pin": out_pin if out_pin else "/Y",  # Default to /Y for ASAP7
      "c_in": c_in_list
    }
    
    cell_name_dict[base_name] = str(idx)
  
  return cell_dict, cell_name_dict


def iterate_nets_get_properties(ord_design, timing, nets, block, cell_dict, cell_name_dict):
  """
  Traverses the netlist and extracts circuit properties from OpenROAD to build a graph.
  
  How data is obtained from OpenROAD:
  1. Iterates through all nets using block.getNets()
  2. For each net, gets connected pins using net.getITerms()
  3. Extracts instance properties using OpenDB APIs:
     - inst.getMaster() for cell type
     - master.getWidth()/getHeight() for area
  4. Extracts timing properties using pin_properties() which calls OpenSTA APIs
  5. Builds connectivity graph (srcs, dsts) for fanin/fanout relationships
  
  Args:
    ord_design: OpenROAD Design object
    timing: OpenROAD Timing object (OpenSTA)
    nets: List of all nets from block.getNets()
    block: Design block containing instances
    cell_dict: Dictionary of cell types
    cell_name_dict: Lookup table for cell names
    
  Returns:
    tuple: (inst_dict, endpoints, srcs, dsts, fanin_dict, fanout_dict)
           - inst_dict: Properties of all instances (slack, slew, load, area)
           - endpoints: List of endpoint nodes (flip-flops)
           - srcs, dsts: Graph connectivity (source and destination indices)
           - fanin_dict, fanout_dict: Fanin/fanout relationships
  """
  #This must eventually be put into a create graph function.
  #source and destination instances for the graph function.
  srcs = []
  dsts = []
  #Dictionary that stores all the properties of the instances.
  inst_dict = {}
  #Dictionary that keeps a stores the fanin and fanout of the instances in an easily indexable way.
  fanin_dict = {}
  fanout_dict = {}
  #storing all the endpoints(here they are flipflops)
  endpoints = []
  
  # Traverse all nets using OpenDB API
  print(f"Starting netlist traversal...")
  total_nets = len(list(nets))
  print(f"Total nets to process: {total_nets}")
  
  for net_idx, net in enumerate(nets):
    # Progress update every 5000 nets
    if net_idx > 0 and net_idx % 5000 == 0:
      print(f"Processed {net_idx}/{total_nets} nets, {len(inst_dict)} instances so far...")
    # Get all pins (ITerms) connected to this net
    iterms = net.getITerms()
    net_srcs = []
    net_dsts = []
    
    # create/update the instance dictionary for each net.
    for s_iterm in iterms:
      # Extract instance information using OpenDB APIs
      inst = s_iterm.getInst()
      inst_name = s_iterm.getInst().getName()
      term_name = s_iterm.getInst().getName() + "/" + s_iterm.getMTerm().getName()
      cell_type = s_iterm.getInst().getMaster().getName()  # Library cell type
      
      # Skip special cells that don't have size variants (tapcells, fillers, SRAMs, etc.)
      if any(x in cell_type.upper() for x in ["TAPCELL", "FILLER", "ENDCAP", "SRAM", "FAKERAM"]):
        continue
  
      if inst_name not in inst_dict:
        # Get instance and master (library cell) from OpenDB
        i_inst = block.findInst(inst_name)
        m_inst = i_inst.getMaster()
        # Calculate area from physical dimensions
        area = m_inst.getWidth() * m_inst.getHeight()
        inst_dict[inst_name] = {
          'idx':len(inst_dict),
          'cell_type_name':cell_type,
          'cell_type':get_type(cell_type, cell_dict, cell_name_dict),
          'slack':0,
          'slew':0,
          'load':0,
          'cin':0,
          'area': area}
      if s_iterm.isInputSignal():
        # Input pin: this is a destination (sink) in the graph
        net_dsts.append((inst_dict[inst_name]['idx'],term_name))
        if inst_dict[inst_name]['cell_type'][0] == 16: # check for flipflops
          endpoints.append(inst_dict[inst_name]['idx'])
      elif s_iterm.isOutputSignal():
        # Output pin: this is a source (driver) in the graph
        net_srcs.append((inst_dict[inst_name]['idx'],term_name))
        # Extract timing properties from OpenROAD using pin_properties()
        # This calls OpenSTA APIs: getPinSlack(), getPinSlew(), getPortCap()
        if net_idx > 0 and net_idx % 10000 == 0:
          print(f"  Extracting timing properties at net {net_idx}...")
        (inst_dict[inst_name]['slack'],
         inst_dict[inst_name]['slew'],
         inst_dict[inst_name]['load'])= pin_properties(s_iterm, CLKset, ord_design, timing)
      # else: Pin is neither input nor output (power, clock, etc.) - skip it
    # list the connections for the graph creation step and the fainin/fanout dictionaries
    for src,src_term in net_srcs:
      for dst,dst_term in net_dsts:
        srcs.append(src)
        dsts.append(dst)
        src_key = list(inst_dict.keys())[src]
        dst_key = list(inst_dict.keys())[dst]
        if src_key in fanout_dict.keys():
          fanout_dict[src_key].append(dst_key)
        else:
          fanout_dict[src_key] = [dst_key]
        if dst_key in fanin_dict.keys():
          fanin_dict[dst_key].append(src_key)
        else:
          fanin_dict[dst_key] = [src_key]
  
  print(f"Completed netlist traversal!")
  print(f"Total instances: {len(inst_dict)}")
  print(f"Total endpoints (flip-flops): {len(endpoints)}")
  print(f"Total graph edges: {len(srcs)}")
  return inst_dict, endpoints, srcs, dsts, fanin_dict, fanout_dict


