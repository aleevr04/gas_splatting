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
    quiet = cfg.quiet

    set_publication_style()

    if not quiet: print(f"Using device: {cfg.device}")

    # --- Simulation ---
    batch, environment = generate_simulation_data(cfg)

    # --- Initialization ---
    t0 = time.time()
    model, init_data = setup_gs_model(batch, cfg)
    setup_time = time.time() - t0
    if not quiet: print(f"Model initialized with {model.num_gaussians} Gaussians in {setup_time:.3f}s")

    plot_initial_guess(environment.ground_truth.gas_map, init_data, cfg)

    # --- Training ---
    trainer = Trainer(model, cfg, environment=environment)

    if not quiet: print("Starting Gas Splatting training...")
    trainer.train(batch)
    results = trainer.finish()
    if not quiet: print(f"Loss: {results.loss_history[-1]:.6f}")
    if not quiet: print(f"Setup Time: {setup_time:.3f}s | Training Time: {results.training_time:.3f}s")

    # --- Plot Results ---
    plot_training_results(model, batch, environment, results, cfg)

    # --- Save Results ---
    metadata = {
        "experiment_name": "training",
        "num_beams": cfg.sim.num_beams,
        "map_size": cfg.env.map_size,
        "cell_size": cfg.env.cell_size
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