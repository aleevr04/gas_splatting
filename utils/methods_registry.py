import time
import utils.tomo_utils as tm
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.plot_utils import render_gaussian_map

def run_sart(system_matrix, measurements, sim_data, cfg, setup_time=0.0):
    t0 = time.time()
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    res = tm.sart(system_matrix, measurements, grid_size=(grid_h, grid_w), num_iterations=400, relaxation_factor=1.6)
    recon_time = time.time() - t0
    return res, setup_time + recon_time

def run_rbf_sart(system_matrix, measurements, sim_data, cfg, setup_time=0.0):
    t0 = time.time()
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    res = tm.rbf_sart(system_matrix, measurements, grid_size=(grid_h, grid_w), cell_size_m=cfg.sim.cell_size)
    recon_time = time.time() - t0
    return res, setup_time + recon_time

def run_lfd(system_matrix, measurements, sim_data, cfg, setup_time=0.0):
    t0 = time.time()
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    res = tm.lfd(system_matrix, measurements, grid_size=(grid_h, grid_w), alpha=0.07)
    recon_time = time.time() - t0
    return res, setup_time + recon_time

def run_ltd(system_matrix, measurements, sim_data, cfg, setup_time=0.0):
    t0 = time.time()
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    res = tm.ltd(system_matrix, measurements, grid_size=(grid_h, grid_w), alpha=5.0)
    recon_time = time.time() - t0
    return res, setup_time + recon_time

def run_gas_splatting(system_matrix, measurements, sim_data, cfg, setup_time=0.0):
    t_start = time.time()
    
    model, _, _ = setup_gs_model(sim_data, cfg)
    trainer = Trainer(model, cfg)
    trainer.train(sim_data)
    gs_img = render_gaussian_map(model, cfg.sim.map_size, cfg.device, cell_size=cfg.sim.cell_size)
    
    total_time = time.time() - t_start
    return gs_img, total_time

AVAILABLE_METHODS = {
    "SART": {
        "func": run_sart,
        "style": {"color": "tab:orange", "marker": "o"}
    },
    "RBF Coupled SART": {
        "func": run_rbf_sart,
        "style": {"color": "tab:green", "marker": "s"}
    },
    "LFD": {
        "func": run_lfd,
        "style": {"color": "tab:red", "marker": "^"},
    },
    "LTD": {
        "func": run_ltd,
        "style": {"color": "tab:purple", "marker": "v"}
    },
    "Gas Splatting": {
        "func": run_gas_splatting,
        "style": {"color": "tab:blue", "marker": "*", "linewidth": 3, "markersize": 12}
    }
}