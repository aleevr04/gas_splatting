import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

from config import Config
from gs_model import GasSplattingModel
from trainer import Trainer
from utils.init_utils import lsqr_initialization
from utils.plot_utils import plot_initial_guess, plot_final_results
from utils.sim_utils import generate_simulation_data


def main():
    # ------- Configuration ------
    parser = ArgumentParser(description="Gas Splatting parameters")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg

    print(f"Using device: {cfg.device}")

    # ------- Simulation ------
    sim_data = generate_simulation_data(cfg)

    # -------- Initialize gaussians --------
    print(f"Running Least Squares initialization (Grid {cfg.init.coarse_res}x{cfg.init.coarse_res})")
    init_pos, init_concentration, init_std, img_coarse = lsqr_initialization(
        sim_data.beams, 
        sim_data.y_true, 
        cfg.sim.map_size, 
        num_gaussians=cfg.init.initial_gaussians,
        coarse_res=cfg.init.coarse_res
    )
    initial_gaussians = init_pos.shape[0]

    model = GasSplattingModel(initial_gaussians, cfg).to(cfg.device)
    model.initialize_gaussians(
        init_pos.to(cfg.device), 
        init_concentration.to(cfg.device), 
        init_std
    )
    print("Model initialized")

    plot_initial_guess(img_coarse, init_pos, cfg.sim.map_size)

    # -------- Training ----------
    trainer = Trainer(model, cfg)

    print("Starting Gas Splatting training...")
    loss_history = trainer.train(sim_data.p_rays, sim_data.u_rays, sim_data.y_true)
    print(f"Loss: {loss_history[-1]:.6f}")

    # --------- Plot Results --------
    plot_final_results(model, sim_data, loss_history, cfg)

if __name__ == "__main__":
    main()