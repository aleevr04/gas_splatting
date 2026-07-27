import time
from simple_parsing import ArgumentParser

from config import Config
from trainer import Trainer
from utils.init_utils import setup_gs_model
from utils.plot_utils import plot_initial_guess, plot_training_results, set_publication_style
from utils.sim_utils import generate_simulation_data
from utils.experiment_utils import save_experiment_results

def main():
    # --- Configuration ---
    parser = ArgumentParser(description="Gas Splatting parameters")
    parser.add_arguments(Config, dest="cfg")
    args = parser.parse_args()
    cfg: Config = args.cfg

    set_publication_style()

    print(f"Using device: {cfg.device}")

    # --- Simulation ---
    sim_data = generate_simulation_data(cfg)

    # --- Initialization ---
    print(f"Running Least Squares initialization...")
    t0 = time.time()
    model, init_pos, img_coarse = setup_gs_model(sim_data.batch, cfg)
    setup_time = time.time() - t0
    print(f"Model initialized with {model.num_gaussians} Gaussians in {setup_time:.3f}s")

    plot_initial_guess(sim_data.ground_truth, img_coarse, init_pos, cfg)

    # --- Training ---
    trainer = Trainer(model, cfg, ground_truth=sim_data.ground_truth)

    print("Starting Gas Splatting training...")
    trainer.train(sim_data.batch)
    results = trainer.finish()
    print(f"Loss: {results.loss_history[-1]:.6f}")
    print(f"Setup Time: {setup_time:.3f} | Training Time: {results.training_time:.3f}")

    # --- Plot Results ---
    plot_training_results(model, sim_data, results, cfg)

    # --- Save Results ---
    metadata = {
        "experiment_name": "training",
        "num_beams": cfg.sim.num_beams,
        "map_size": cfg.sim.map_size,
        "cell_size": cfg.sim.cell_size
    }

    results = {
        "initial_gaussians": model.initial_gaussians,
        "final_gaussians": model.num_gaussians,
        "setup_time": setup_time,
        "training_time": results.training_time,
        "loss_history": results.loss_history,
        "rmse_history": results.rmse_history,
        "densify_history": results.densify_history
    }

    save_experiment_results(metadata, results)

if __name__ == "__main__":
    main()