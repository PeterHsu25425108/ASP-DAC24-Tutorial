# Quick Reference: Critical Path Finding and OpenROAD Data Extraction

## Question 1: How are critical paths found?

### Answer:
Critical paths are found using the `get_critical_path_nodes()` function (line 262 in `demo2_gate_sizing_helpers.py`):

1. **Uses `torch.topk()`** to find nodes with the smallest (most negative) slack values
2. **Dynamic scope adjustment**: Number of critical nodes = `TOP_N_NODES * (1 + 0.01 * episode_num)`
   - Starts with 100 nodes in episode 0
   - Grows to 200 nodes by episode 100
3. **Filters negative slacks**: Only keeps nodes with timing violations (slack < 0)
4. **Fallback**: If no violations, returns all nodes for area optimization

```python
# Example: Finding critical nodes
topk = min(len(graph.ndata['slack'])-1, int(100*(1+0.01*ep_num)))
min_slacks, critical_path = torch.topk(graph.ndata['slack'], topk, largest=False)
critical_path = critical_path[min_slacks < 0]  # Keep only negative slacks
```

## Question 2: How is data obtained from OpenROAD?

### Answer:
Data is obtained through **OpenROAD Python APIs** (no file I/O needed):

### Key Functions and APIs:

#### 1. Design Loading (`load_design()` - line 557)
```python
# Load technology and netlist
ord_tech.readLiberty(lib_file)      # Timing characterization
ord_tech.readLef(lef_file)          # Physical layout info
ord_design.readVerilog(netlist)      # Load netlist
timing = Timing(ord_design)          # Create OpenSTA timing engine
```

#### 2. Timing Data Extraction (`pin_properties()` - line 113)
```python
# Get slack (timing margin)
slack = timing.getPinSlack(dbpin, timing.Fall, timing.Max)

# Get slew (transition time)
slew = timing.getPinSlew(dbpin)

# Get load capacitance
load = timing.getPortCap(ITerm, corner, timing.Max)
```

#### 3. Graph Building (`iterate_nets_get_properties()` - line 617)
```python
# Traverse netlist
for net in block.getNets():
    iterms = net.getITerms()          # Get pins
    inst = iterm.getInst()            # Get instance
    cell_type = inst.getMaster()      # Get cell type
    area = master.getWidth() * master.getHeight()  # Physical area
```

#### 4. Real-time Updates During Optimization (`env_step()` - line 258)
```python
# Resize gate
inst.swapMaster(new_master)           # Change gate size
# OpenROAD automatically recalculates timing!
new_slack = timing.getPinSlack(dbpin) # Get updated slack
```

### Data Flow Summary:

```
OpenROAD Libraries → load_design() → Tech/Design/Timing objects
                              ↓
Netlist Traversal → iterate_nets_get_properties() → Extract timing data
                              ↓                       (slack, slew, load)
Build DGL Graph ← Store in graph.ndata['slack'] ← via timing.getPinSlack()
                              ↓
Training Loop → get_critical_path_nodes() → Find worst slack nodes
                              ↓
RL Agent Action → env_step() → inst.swapMaster() → Auto timing update
                              ↓
New Timing Data ← timing.getPinSlack() ← OpenSTA recalculates
```

## Key Benefits of OpenROAD Python APIs:

1. **No File I/O**: Direct in-memory data access (much faster than TCL)
2. **Real-time Updates**: Timing recalculated automatically after gate swaps
3. **Incremental Analysis**: OpenSTA updates only affected paths
4. **Easy Integration**: Works seamlessly with PyTorch and DGL

## For More Details:

See `CRITICAL_PATH_DOCUMENTATION.md` in this directory for comprehensive explanation with code examples and diagrams.
