from simple_parsing import ArgumentParser

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.plot_utils import plot_initial_guess, plot_training_results
from utils.sim_utils import generate_simulation_data


def main():
    # --- Configuration ---
    parser = ArgumentParser(description="Gas Splatting parameters")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg

    print(f"Using device: {cfg.device}")

    # --- Simulation ---
    sim_data = generate_simulation_data(cfg)

    # --- Initialization ---
    print(f"Running Least Squares initialization (coarse_cell_size = {cfg.init.coarse_cell_size})...")
    model, init_pos, img_coarse = setup_gs_model(sim_data, cfg)
    print("Model initialized")

    plot_initial_guess(sim_data.img_gt, img_coarse, init_pos, cfg)

    # --- Training ---
    trainer = Trainer(model, cfg)

    print("Starting Gas Splatting training...")
    loss_history = trainer.train(sim_data.beams, sim_data.measurements)
    print(f"Loss: {loss_history[-1]:.6f}")

    # --- Plot Results ---
    plot_training_results(model, sim_data, loss_history, cfg)

if __name__ == "__main__":
    main()