# Generating DG Data using Simbench and Powerdata-gen
#
# This script was modified from the data generation pipeline used in
# the ENGAGE project (https://gitlab.lrz.de/energy-management-technologies-public/engage/-/blob/main/graph_gen.py).

### Load dependencies, including the data generator library

import sys, os
DATA_GEN_PATH = os.path.abspath('powerdata-gen/')
sys.path.append(DATA_GEN_PATH)
import powerdata_gen
# Reset path
sys.path.pop()

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.networks import create_cigre_network_lv, create_kerber_dorfnetz
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm

import time
import logging
import argparse
from copy import deepcopy
import random

### Helper functions for script arguments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry_run",
        action="store_true"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


### Helper functions to load pandapower grids from simbench

def create_output_dir():
    identifier = time.strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join('outputs', identifier)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_cigre_network():
    net = create_cigre_network_lv()
    # Standard CIGRE LV DER Configuration
    # We use sgen (static generator) for PV and Fuel Cells
    der_config = [
        # Photovoltaics (Standard sgen)
        {"bus": "Bus R11", "p": 0.004, "type": "PV", "cat": "sgen"},
        {"bus": "Bus R15", "p": 0.003, "type": "PV", "cat": "sgen"},
        {"bus": "Bus R18", "p": 0.003, "type": "PV", "cat": "sgen"},
        {"bus": "Bus I2", "p": 0.030, "type": "PV", "cat": "sgen"},
        {"bus": "Bus C12", "p": 0.010, "type": "PV", "cat": "sgen"},
        {"bus": "Bus C20", "p": 0.010, "type": "PV", "cat": "sgen"},
        # Fuel Cell (Standard sgen, usually type 'CHP')
        {"bus": "Bus R18", "p": 0.005, "type": "CHP", "cat": "sgen"},
        # Battery (Proper storage element)
        {"bus": "Bus R17", "p": 0.005, "type": "Battery", "cat": "storage", "max_e": 0.007}
    ]

    for der in der_config:
        b_idx = net.bus[net.bus.name == der['bus']].index[0]

        if der['cat'] == "sgen":
            pp.create_sgen(net, b_idx, p_mw=der['p'], q_mvar=0, type=der['type'])
        elif der['cat'] == "storage":
            pp.create_storage(net, b_idx, p_mw=der['p'], max_e_mwh=der['max_e'], q_mvar=0, type=der['type'])

    return net

def create_future_kerber_dorfnetz(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Load the base topology (Village grid: 57 loads, 116 buses)
    net = create_kerber_dorfnetz()
    
    # 2. Get list of all load buses (potential connection points)
    load_buses = net.load.bus.values.tolist()
    n_households = len(load_buses)

    # 3. Apply heretogenous base loads so not all constant
    # We redefine the initial 6kW baseline with three classes:
    # Small (0.6x), Standard (1.0x), and Heavy (2.5x)
    for i in net.load.index:
        size_factor = np.random.choice([0.6, 1.0, 2.5], p=[0.3, 0.5, 0.2])
        net.load.at[i, "p_mw"] *= size_factor
        net.load.at[i, "name"] = f"Base_Load_{size_factor*6}kW"
    
    # 4. Define Penetration Counts
    n_pv = int(0.40 * n_households)
    n_ev = int(0.20 * n_households)
    n_hp = int(0.15 * n_households)
    
    # 5. Randomly select buses for each technology
    pv_buses = random.sample(load_buses, n_pv)
    ev_buses = random.sample(load_buses, n_ev)
    hp_buses = random.sample(load_buses, n_hp)
    
    # 6. Add Photovoltaics (sgen)
    for bus in pv_buses:
        p_val = random.uniform(0.005, 0.015) # 5-15 kW
        pp.create_sgen(net, bus, p_mw=p_val, q_mvar=0, type='PV', name=f"PV_at_{bus}")
        
        # 7. Add Batteries (30% of PV sites)
        if random.random() < 0.30:
            pp.create_storage(net, bus, p_mw=0.005, max_e_mwh=0.010, q_mvar=0, 
                              type='Battery', name=f"BESS_at_{bus}")

    # 8. Add Electric Vehicles (sgen or load - using sgen with negative P for flexibility)
    for bus in ev_buses:
        # 11 kW Wallbox. Negative P in sgen = consumption (load convention)
        pp.create_sgen(net, bus, p_mw=-0.011, q_mvar=0, type='EV', name=f"EV_at_{bus}")

    # 9. Add Heat Pumps (separate load elements)
    for bus in hp_buses:
        p_val = random.uniform(0.003, 0.006) # 3-6 kW
        pp.create_load(net, bus, p_mw=p_val, q_mvar=0, type='HeatPump', name=f"HP_at_{bus}")

    return net


def save_pandapower_grid_to_json(grid_code: str, filename: str):
    if grid_code == 'CIGRE_LV':
        net = get_cigre_network()
    elif grid_code == 'Kerber_Dorfnetz':
        net = create_future_kerber_dorfnetz()
    else:
        raise ValueError(f'Unsupported grid code: {grid_code}')

    pp.to_json(net, filename)
    return filename


### Helper functions for extracting node features and edge features

def get_node_features(net):
    # List of bus features
    #   x: np.array([Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree])
    #   y: np.array([p_mw, q_mvar, vm_pu, va_degree])
    #
    node_features_x, node_features_y = [], [] # map from bus_id to features
    for bus_id in net.bus.index:
        # (Slack?, PV?, PQ?)
        bus_type = (0, 0, 1)

        gens = net.gen.loc[net.gen['bus'] == bus_id]
        if len(gens) > 0:
            bus_type = (0, 1, 0)

        slack = net.ext_grid.loc[net.ext_grid['bus'] == bus_id,
                        ['vm_pu', 'va_degree']]
        if len(slack) > 0:
            assert len(gens) == 0, ('PV and Swing generators cannot be placed'
                                    ' on the same bus. This is because they'
                                    ' will both try to control the bus voltage.')
            bus_type = (1, 0, 0)
        
        # net.res_bus should already take into account all the components that
        # contribute to these four bus parameters so we do not have to do this
        # again (ex. loads, sgens, gens, storages, ext_grid, etc.).
        features = net.res_bus.loc[bus_id, ['p_mw', 'q_mvar', 'vm_pu', 'va_degree']]
        masked_features = features.copy()
        if bus_type[0]:
            masked_features['p_mw'] = np.nan
            masked_features['q_mvar'] = np.nan
        elif bus_type[1]:
            masked_features['q_mvar'] = np.nan
            masked_features['va_degree'] = np.nan
        else:
            masked_features['vm_pu'] = np.nan
            masked_features['va_degree'] = np.nan

        node_features_x.append(np.append(bus_type, masked_features.values))
        node_features_y.append(features.values)
    
    return np.array(node_features_x), np.array(node_features_y)

def get_edge_features(net):
    # List of edge features
    #   e: np.array([trafo?, r_pu, x_pu, sc_voltage])

    def get_line_features(net):
        # Undirected graph so need to add both directions to edge_index.
        edge_index = net.line.loc[:, ['from_bus', 'to_bus',
                                      'to_bus', 'from_bus']].values
        # Use .reshape to change shape from (E, 4) to (2E, 2), where E is num edges.
        # Transpose to make into proper (2, 2E format).
        edge_index = edge_index.reshape(-1, 2).T

        r = net.line['r_ohm_per_km'].values * net.line['length_km'].values
        x = net.line['x_ohm_per_km'].values * net.line['length_km'].values

        # We convert the r,x values into per unit (p.u.) to simplify calculations
        # and ensure consistency across the network. To do this, we divide r, x by
        # the base impedance. Therefore z = vn_kv**2/sn_mva, where vn_kv is rated
        # voltage and sn_mva is reference apparent power.
        # Note: vn_kv be the same for every bus except ext_grid, but this is safer.
        vn_kv = net.bus.loc[net.line['to_bus'], ['vn_kv']].values.reshape(-1)
        z = np.square(vn_kv) / net.sn_mva
        r_pu = r / z
        x_pu = x / z

        # Similarly, due to undirected graph, the edge features need to be repeated
        # twice, once for each respective connection present in the COO matrix.
        r_pu = r_pu.repeat(2)
        x_pu = x_pu.repeat(2)

        # Add encoding for a line and pad with nan to account for missing short
        # circuit voltage.
        e = edge_index.shape[1] # b/c coo matrix
        edge_features = np.vstack([np.zeros(e),         # trafo?
                                   r_pu,                # r_pu
                                   x_pu,                # x_pu
                                   np.nan*np.ones(e)    # sc_voltage
                                   ]).T

        return edge_index, edge_features

    def get_trafo_features(net):
        # Similar to get_line_features.
        edge_index = net.trafo.loc[:, ['hv_bus', 'lv_bus',
                                       'lv_bus', 'hv_bus']].values
        edge_index = edge_index.reshape(-1, 2).T

        # Impedance calculated as shown in pandapower docs:
        # https://pandapower.readthedocs.io/en/v2.14.11/elements/trafo.html#impedance-values
        # where vk_percent is short-circuit voltage and vkr_percent is the real
        # part of short-circuit voltage (%).
        z_pu = (net.trafo['vk_percent'].values / 100)*(net.sn_mva / net.trafo['sn_mva'].values)
        r_pu = (net.trafo['vkr_percent'].values / 100)*(net.sn_mva / net.trafo['sn_mva'].values)
        x_pu = np.sqrt(np.square(z_pu) - np.square(r_pu))

        # Add relative short-circuit voltage as additional feature.
        sc_voltage = net.trafo['vk_percent'].values
        
        # Repeat the features (to match edge_index) and create feature matrix.
        e = edge_index.shape[1] # b/c coo matrix
        edge_features = np.vstack([np.ones(e),              # trafo?
                                   r_pu.repeat(2),          # r_pu
                                   x_pu.repeat(2),          # x_pu
                                   sc_voltage.repeat(2)     # sc_voltage
                                   ]).T

        return edge_index, edge_features
    
    A_line, E_line = get_line_features(net)
    A_trafo, E_trafo = get_trafo_features(net)
    
    # Combine and return the line and trafo features.
    A = np.hstack([A_line, A_trafo])
    E = np.vstack([E_line, E_trafo])

    # Sometimes bus ids are higher than the number of nodes. This can mess up
    # the adjacency matrix (edge_index) so we need to remap back to smaller ids.
    # We assume the graph is fully connected, so every node id exists at least
    # once in the edge_index.
    unique_nodes = set(A[0])
    remapping = dict(zip(sorted(unique_nodes), range(len(unique_nodes))))
    applyall = np.vectorize(lambda x: remapping[x])
    A = applyall(A)
    
    return A, E

def convert_switches_to_lines(net):
    """
    Replaces closed bus-to-bus switches with lines using the average
    characteristics of the existing network.
    """
    # 1. Calculate averages from your specific net.line table
    # This ensures the 'synthetic' lines look like the 'real' lines to the GNN
    avg_len = net.line.length_km.mean()
    avg_r = net.line.r_ohm_per_km.mean()
    avg_x = net.line.x_ohm_per_km.mean()
    avg_i = net.line.max_i_ka.mean()

    # 2. Extract closed bus-to-bus switches
    bus_sw = net.switch[(net.switch.closed == True) & (net.switch.et == 'b')].copy()

    for _, sw in bus_sw.iterrows():
        # Create a line using the grid's average parameters
        pp.create_line_from_parameters(
            net, 
            from_bus=sw.bus, 
            to_bus=sw.element, 
            length_km=avg_len,
            r_ohm_per_km=avg_r, 
            x_ohm_per_km=avg_x,
            c_nf_per_km=0, # Switches typically have no capacitance
            max_i_ka=avg_i,
            name=f"Switch_Line_{sw.name if sw.name else sw.index}"
        )

    # 3. Handle Breakers (Lines and Transformers)
    # If a breaker (switch type 'l' or 't') is OPEN, 
    # we just mark that element as out of service.
    for et, table in [('l', net.line), ('t', net.trafo)]:
        open_ids = net.switch[(net.switch.closed == False) & (net.switch.et == et)].element.values
        table.loc[table.index.isin(open_ids), 'in_service'] = False

    # 4. Remove all switches
    # Now all connectivity is stored in net.line and net.trafo
    net.switch.drop(net.switch.index, inplace=True)
    
    return net

def verify_connectivity(net):
    # 1. Initialize an undirected graph
    G = nx.Graph()
    
    # 2. Add all buses as nodes
    # Using the full index ensures we account for potentially isolated nodes
    G.add_nodes_from(net.bus.index)
    
    # 3. Add edges from lines that are in service
    lines = net.line[net.line.in_service == True]
    for _, row in lines.iterrows():
        G.add_edge(row.from_bus, row.to_bus)
        
    # 4. Add edges from transformers that are in service
    trafos = net.trafo[net.trafo.in_service == True]
    for _, row in trafos.iterrows():
        G.add_edge(row.hv_bus, row.lv_bus)
        
    # 5. Perform Connectivity Analysis
    is_connected = nx.is_connected(G)
    
    if not is_connected:
        print("\nWarning: The following components are isolated:")
        components = list(nx.connected_components(G))
        for i, comp in enumerate(components):
            print(f" Component {i+1} (Size {len(comp)}): {list(comp)[:10]}...") 
            # (Truncated list for brevity)

    return is_connected

def get_pyg_data_from_net(net):
    net = convert_switches_to_lines(net)
    is_connected = verify_connectivity(net)
    if not is_connected:
        return None

    X_i, Y_i = get_node_features(net)
    A_i, E_i = get_edge_features(net)

    # Run dc_pf, such that we can use this later and do not have to compute every time.

    # Load the source network
    net = deepcopy(net)
    # Run dc pf
    pp.rundcpp(net)
    # Put this in correct format to match the true data and get np array.
    np_dc_pf = net.res_bus[['p_mw', 'q_mvar', 'vm_pu', 'va_degree']].values
    # Convert to tensor and replace nan (q_mwar) with 0.
    dc_pf = torch.nan_to_num(torch.Tensor(np_dc_pf), nan=0.0)

    # Data dimensions
    #   x: (N, 7), where 7 are [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]
    #   edge_index: (2, 2E)
    #   edge_attr: (2E, 4), where 4 are [trafo?, r_pu, x_pu, sc_voltage]
    #   y: (N, 4), where 4 are [p_mw, q_mvar, vm_pu, va_degree]
    #   dc_pf: (N, 4), where 4 are [p_mw, q_mvar, vm_pu, va_degree]
    #
    return Data(x=torch.tensor(X_i, dtype=torch.float32),
                edge_index=torch.tensor(A_i, dtype=torch.int64),
                edge_attr=torch.tensor(E_i, dtype=torch.float32),
                y=torch.tensor(Y_i, dtype=torch.float32),
                dc_pf=dc_pf)

### Generate Grids

if __name__ == '__main__':
    args = parse_args()

    TEST_GENERATION = args.dry_run
    size = args.size

    # Setup input directory
    input_dir = 'inputs/'
    os.makedirs(input_dir, exist_ok=True)

    # Convert simbench codes to json files, if not already done.
    filenames = []
    grid_codes = ['CIGRE_LV', 'Kerber_Dorfnetz']
    for code in grid_codes:
        f = os.path.join(input_dir, f'{code}.json')
        if not os.path.exists(f):
            save_pandapower_grid_to_json(code, f)
        filenames.append(f)

    # Load a base config file and change adjust parameters.
    cfg = OmegaConf.load('base_gen_config.yaml')
    cfg.n_train, cfg.n_val, cfg.n_test = dataset_split
    cfg.seed = 12

    # Set up logger (for powerdata-gen)
    log = logging.getLogger(__name__)

    # Create output directory for all data from this run, loop through ref grids, 
    # generate new grids for each ref, save them to subdir of output dir.
    output_dir = create_output_dir()
    print(f'Output directory: {output_dir}\n')
    generated_grid_base_dirs = []
    for code, f in list(zip(grid_codes, filenames)):
        save_path = os.path.join(output_dir, code)
        os.makedirs(save_path, exist_ok=True)
        cfg.default_net_path = f
        powerdata_gen.build_datasets(cfg.default_net_path,
                                     save_path,
                                     log,
                                     cfg.n_train,
                                     cfg.n_val,
                                     cfg.n_test,
                                     cfg.keep_reject,
                                     cfg.sampling,
                                     cfg.powerflow,
                                     cfg.filtering,
                                     cfg.seed)
        generated_grid_base_dirs.append(save_path)


    # Create PyTorch datasets using the generated grids

    for dir in generated_grid_base_dirs:
        generated_grid_dir = os.path.join(dir, 'train')
        generated_grids = os.listdir(generated_grid_dir)
        # list[outputs/<identifier>/<sb_code>/<train|test|val>/sample_<N>.json]
        generated_grid_files = [os.path.join(generated_grid_dir, f) for f in generated_grids]

        dataset_filename = os.path.join(generated_grid_dir,
                                        f'dataset.pt')
        
        dataset_source = os.path.join(generated_grid_dir,
                                        f'dataset_src.csv')

        # If we have already created this dataset, skip.
        if os.path.exists(dataset_filename) and os.path.exists(dataset_source):
            continue
        dataset = []
        srcs = []

        for f in tqdm(generated_grid_files):
            if f.split('.')[-1] != 'json':
                # There could be non json files that exist, so skip them.
                continue
            net = pp.from_json(f)
            data = get_pyg_data_from_net(net)
            if data is None:
                continue
            ppci = net["_ppc"]["internal"] # In some cases, may need to run `pp.runpp(net, verbose=False)` right above this.
            data.ppci = ppci

            dataset.append(data)
            srcs.append(f)
        
        print('Saving dataset in', dataset_filename, end='... ')
        torch.save(dataset, dataset_filename)
        print('completed')
        print('Saving source list in', dataset_source,  end='... ')
        pd.DataFrame(srcs, columns=['src']).to_csv(dataset_source)
        print('completed')




