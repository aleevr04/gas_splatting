import time
import numpy as np
from scipy.sparse import csr_matrix

import utils.tomo_utils as tm
from utils.init_utils import setup_gs_model
from utils.sim_utils import MeasurementBatch
from config import Config
from trainer import Trainer

def get_tomo_inputs(batch: MeasurementBatch, cfg: Config):
    """Helper to prevent repeating grid math and tensor conversions."""
    grid_w = int(cfg.sim.map_size[0] / cfg.sim.cell_size)
    grid_h = int(cfg.sim.map_size[1] / cfg.sim.cell_size)
    measurements = batch.measurements.cpu().numpy()
    return (grid_h, grid_w), measurements

def run_sart(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, matrix_setup_time: float, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)

    t0 = time.time()
    res = tm.sart(system_matrix, measurements, grid_size=grid_size, num_iterations=400, relaxation_factor=1.6)
    recon_time = time.time() - t0
    return res, matrix_setup_time + recon_time

def run_rbf_sart(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, matrix_setup_time: float, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)

    t0 = time.time()
    res = tm.rbf_sart(system_matrix, measurements, grid_size=grid_size, cell_size_m=cfg.sim.cell_size)
    recon_time = time.time() - t0
    return res, matrix_setup_time + recon_time

def run_lfd(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, matrix_setup_time: float, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)

    t0 = time.time()
    res = tm.lfd(system_matrix, measurements, grid_size=grid_size, alpha=0.07)
    recon_time = time.time() - t0
    return res, matrix_setup_time + recon_time

def run_ltd(batch: MeasurementBatch, cfg: Config, system_matrix: csr_matrix, matrix_setup_time: float, **kwargs):
    grid_size, measurements = get_tomo_inputs(batch, cfg)
    
    t0 = time.time()
    res = tm.ltd(system_matrix, measurements, grid_size=grid_size, alpha=5.0)
    recon_time = time.time() - t0
    return res, matrix_setup_time + recon_time

def run_gas_splatting(batch: MeasurementBatch, cfg: Config, ground_truth: np.ndarray | None = None, **kwargs):
    t_start = time.time()
    model, _, _ = setup_gs_model(batch, cfg)
    gs_setup_time = time.time() - t_start

    trainer = Trainer(model, cfg, ground_truth)
    trainer.train(batch)
    results = trainer.finish()
    gs_img = model.render_map(cell_size=cfg.sim.cell_size)
    
    return gs_img, gs_setup_time + results.training_time

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