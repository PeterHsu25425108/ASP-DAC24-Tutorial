# Critical Path Finding in Session1's Gate Sizer

This document explains how critical paths are found and how timing data is obtained from OpenROAD in the RL-based gate sizing example (`demo2_gate_sizing.py`).

## Overview

The gate sizer uses a reinforcement learning (RL) approach to optimize gate sizes for timing closure. A key component of this approach is identifying critical paths - the paths with the worst timing slack - and focusing optimization efforts on those paths.

## 1. How Critical Paths Are Found

### Main Function: `get_critical_path_nodes()`

Located in `demo2_gate_sizing_helpers.py` (lines 239-247), this function identifies the nodes (gates) on critical paths:

```python
def get_critical_path_nodes(graph, ep_num, TOP_N_NODES, n_cells):
  topk = min(len(graph.ndata['slack'])-1 , int(TOP_N_NODES*(1+0.01*ep_num)))
  min_slacks, critical_path = torch.topk(graph.ndata['slack'], topk, largest=False)
  critical_path = critical_path[min_slacks<0]

  if critical_path.numel() <=0:
    critical_path = torch.arange(0,graph.num_nodes())

  return critical_path
```

### How It Works:

1. **Dynamic Top-K Selection**: 
   - Calculates `topk` value based on episode number: `TOP_N_NODES * (1 + 0.01 * ep_num)`
   - This gradually increases the number of critical nodes considered as training progresses
   - Default `TOP_N_NODES = 100` (defined in helpers file)

2. **Finding Worst Slack Nodes**:
   - Uses `torch.topk()` with `largest=False` to find nodes with smallest (most negative) slack values
   - Slack values are stored in the graph's node data: `graph.ndata['slack']`

3. **Filtering Only Negative Slacks**:
   - Filters to keep only nodes with negative slack: `critical_path[min_slacks<0]`
   - Negative slack indicates timing violations

4. **Fallback Mechanism**:
   - If no nodes have negative slack (timing is already met), returns all nodes
   - This allows the algorithm to continue optimization for area reduction

### Usage in the Main Training Loop:

In `demo2_gate_sizing.py` (line 226):
```python
critical_nodes = get_critical_path_nodes(episode_G, i_episode, TOP_N_NODES, n_cells)
critical_graph = get_subgraph(episode_G, critical_nodes)
```

The critical nodes are then used to create a subgraph (`critical_graph`) that focuses the RL agent's attention on the most timing-critical parts of the design.

## 2. How Data Is Obtained from OpenROAD

The gate sizer obtains timing and physical data from OpenROAD through its Python APIs. This happens in multiple stages:

### A. Initial Design Loading (`load_design()` function)

Located in `demo2_gate_sizing_helpers.py` (lines 509-527):

```python
def load_design(path):
  ord_tech = Tech()
  lib_file = path/"platforms/lib/NangateOpenCellLibrary_typical.lib"
  lef_file = path/"platforms/lef/NangateOpenCellLibrary.macro.lef"
  tech_lef_file = path/"platforms/lef/NangateOpenCellLibrary.tech.lef"
  
  # Load technology files
  ord_tech.readLiberty(lib_file.as_posix())
  ord_tech.readLef(tech_lef_file.as_posix())
  ord_tech.readLef(lef_file.as_posix())
  
  # Create design and timing objects
  ord_design = Design(ord_tech)
  timing = Timing(ord_design)
  
  # Load netlist
  design_file = path/f"designs/{design}_{semi_opt_clk}.v"
  ord_design.readVerilog(design_file.as_posix())
  ord_design.link(design)
  
  # Set clock constraint
  ord_design.evalTclString("create_clock [get_ports i_clk] -name core_clock -period " + str(clk_init*1e-9))
  
  # Get database objects
  db = ord.get_db()
  chip = db.getChip()
  block = ord.get_db_block()
  nets = block.getNets()
  
  return ord_tech, ord_design, timing, db, chip, block, nets
```

**Key OpenROAD Components Obtained:**
- `Tech`: Technology library information
- `Design`: The design object containing the netlist
- `Timing`: Timing analysis engine (OpenSTA)
- `db`: OpenDB database
- `block`: Design block containing instances and nets

### B. Extracting Pin Properties (`pin_properties()` function)

Located in `demo2_gate_sizing_helpers.py` (lines 113-133):

```python
def pin_properties(dbpin, CLKset, ord_design, timing):
  ITerms = dbpin.getNet().getITerms()
  
  # Get slack from timing engine
  slack = min(timing.getPinSlack(dbpin, timing.Fall, timing.Max), 
              timing.getPinSlack(dbpin, timing.Rise, timing.Max))
  if slack < -0.5*CLKset[0]:
    slack = 0
  
  # Get slew from timing engine
  slew = timing.getPinSlew(dbpin)  
  
  # Calculate total load capacitance
  load = 0
  for ITerm in ITerms:
    if ITerm.isInputSignal():
      new_load = 0
      for corner in timing.getCorners():
        tmp_load = timing.getPortCap(ITerm, corner, timing.Max)
        if tmp_load > new_load:
          new_load = tmp_load
      load += new_load

  return slack, slew, load
```

**OpenROAD Timing APIs Used:**

1. **`timing.getPinSlack(dbpin, rise/fall, min/max)`**
   - Returns the timing slack at a specific pin
   - Calculated as: required_time - arrival_time
   - Negative slack indicates a timing violation

2. **`timing.getPinSlew(dbpin)`**
   - Returns the signal transition time (slew) at the pin
   - Important for delay calculation and library lookup

3. **`timing.getPortCap(ITerm, corner, max/min)`**
   - Returns the input capacitance of a pin
   - Used to calculate the load driven by an output pin

4. **`timing.getCorners()`**
   - Returns all timing corners (e.g., fast, typical, slow)
   - Allows corner-specific timing analysis

### C. Building the Graph (`iterate_nets_get_properties()` function)

Located in `demo2_gate_sizing_helpers.py` (lines 530-592):

This function iterates through all nets in the design and builds a graph representation:

```python
def iterate_nets_get_properties(ord_design, timing, nets, block, cell_dict, cell_name_dict):
  srcs = []
  dsts = []
  inst_dict = {}
  
  for net in nets:
    iterms = net.getITerms()
    
    for s_iterm in iterms:
      inst = s_iterm.getInst()
      inst_name = inst.getName()
      cell_type = inst.getMaster().getName()
  
      if inst_name not in inst_dict:
        i_inst = block.findInst(inst_name)
        m_inst = i_inst.getMaster()
        area = m_inst.getWidth() * m_inst.getHeight()
        
        inst_dict[inst_name] = {
          'idx': len(inst_dict),
          'cell_type_name': cell_type,
          'cell_type': get_type(cell_type, cell_dict, cell_name_dict),
          'slack': 0,
          'slew': 0,
          'load': 0,
          'cin': 0,
          'area': area
        }
      
      # Extract timing properties for output pins
      if s_iterm.isOutputSignal():
        (inst_dict[inst_name]['slack'],
         inst_dict[inst_name]['slew'],
         inst_dict[inst_name]['load']) = pin_properties(s_iterm, CLKset, ord_design, timing)
```

**OpenROAD Database APIs Used:**

1. **`net.getITerms()`**
   - Returns all pins (ITerms) connected to a net

2. **`iterm.getInst()`**
   - Returns the instance (cell) that the pin belongs to

3. **`inst.getMaster()`**
   - Returns the library cell (master) of the instance

4. **`master.getWidth()` / `master.getHeight()`**
   - Returns physical dimensions for area calculation

5. **`iterm.isOutputSignal()` / `iterm.isInputSignal()`**
   - Determines pin direction

### D. Real-Time Updates During Optimization (`env_step()` function)

Located in `demo2_gate_sizing_helpers.py` (lines 258-354):

When the RL agent takes an action (resize a gate), the function updates the design:

```python
def env_step(episode_G, graph, state, action, ...):
  # ... determine which cell to resize ...
  
  # Update the cell master in OpenDB
  inst = block.findInst(inst_name)
  n_master_name = cell_dict[str(cell_idx)]['name'] + cell_dict[str(cell_idx)]['sizes'][cell_size]
  db = ord.get_db()
  n_master = db.findMaster(n_master_name)
  inst.swapMaster(n_master)  # KEY: Swap to new cell size
  
  # Get new timing data from OpenROAD
  dbpin = block.findITerm(inst_name + cell_dict[str(inst_dict[inst_name]['cell_type'][0])]['out_pin'])
  new_slack = min_slack(dbpin, timing)
  
  # Update all affected timing values
  for n, inst in inst_names.items():
    tmp_db_pin = block.findITerm(inst + cell_dict[str(episode_inst_dict[inst]['cell_type'][0])]['out_pin'])
    if tmp_db_pin.getNet() != None:
      (episode_inst_dict[inst]['slack'],
       episode_inst_dict[inst]['slew'],
       episode_inst_dict[inst]['load']) = pin_properties(tmp_db_pin, CLKset, ord_design, timing)
```

**Key OpenROAD Operation:**

- **`inst.swapMaster(n_master)`**: 
  - This is the core operation that changes a gate's size
  - OpenROAD's timing engine automatically recalculates timing after this operation
  - The updated timing values are immediately available through the timing APIs

## 3. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Design (load_design)                                    │
│    - Read Liberty files, LEF files, Verilog netlist             │
│    - Create Tech, Design, Timing objects                        │
│    - Set clock constraints                                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Build Graph (iterate_nets_get_properties)                    │
│    - Traverse all nets using OpenDB APIs                        │
│    - Extract timing data using Timing APIs:                     │
│      * getPinSlack() → slack values                             │
│      * getPinSlew() → slew values                               │
│      * getPortCap() → load capacitance                          │
│    - Build DGL graph with node features                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Training Loop (demo2_gate_sizing.py)                         │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ a. Find Critical Paths (get_critical_path_nodes)         │ │
│    │    - Use torch.topk() on slack values                    │ │
│    │    - Filter negative slacks                              │ │
│    └───────────────────┬─────────────────────────────────────┘ │
│                        │                                         │
│    ┌───────────────────▼─────────────────────────────────────┐ │
│    │ b. Create Subgraph (get_subgraph)                        │ │
│    │    - Extract critical nodes and neighbors                │ │
│    │    - Focus RL agent on critical region                   │ │
│    └───────────────────┬─────────────────────────────────────┘ │
│                        │                                         │
│    ┌───────────────────▼─────────────────────────────────────┐ │
│    │ c. Take Action (env_step)                                │ │
│    │    - Resize gate: inst.swapMaster()                      │ │
│    │    - OpenROAD automatically updates timing               │ │
│    └───────────────────┬─────────────────────────────────────┘ │
│                        │                                         │
│    ┌───────────────────▼─────────────────────────────────────┐ │
│    │ d. Update Timing (pin_properties)                        │ │
│    │    - Query new slack/slew/load from Timing APIs          │ │
│    │    - Calculate reward based on timing improvement        │ │
│    └───────────────────┬─────────────────────────────────────┘ │
│                        │                                         │
│                        └──────────────┐                          │
│                                       │                          │
└───────────────────────────────────────┼──────────────────────────┘
                                        │
                                        └─► Loop continues

```

## 4. Key Insights

### Critical Path Finding Strategy
- **Adaptive Focus**: The number of critical nodes increases with episode number, allowing the agent to gradually expand its optimization scope
- **Negative Slack Priority**: Only nodes with timing violations are considered critical
- **Subgraph Extraction**: The agent works on a subgraph containing critical nodes and their neighbors (2-hop neighborhood), reducing computational complexity

### OpenROAD Integration Benefits
- **No File I/O**: All data exchange happens through in-memory Python APIs, making it much faster than TCL-based approaches
- **Real-time Updates**: Gate sizing changes are immediately reflected in timing calculations
- **Incremental Analysis**: OpenSTA incrementally updates timing, avoiding full re-analysis after each change
- **Direct Database Access**: Can query and modify the design database directly through OpenDB APIs

## 5. Parameters

Key parameters controlling critical path finding:

- **`TOP_N_NODES = 100`**: Base number of critical nodes to consider
- **`ep_num`**: Episode number, used to gradually increase the focus region
- **Growth rate**: `1 + 0.01 * ep_num` means the number of nodes increases by 1% per episode

## References

- OpenROAD Python API documentation: https://github.com/The-OpenROAD-Project/OpenROAD
- Main implementation: `session1/demo2_gate_sizing.py`
- Helper functions: `session1/demo2_gate_sizing_helpers.py`
