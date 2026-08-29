import copy
import os
import sys
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from simple_parsing import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from trainer import Trainer
from utils.experiment_utils import rmse_loss, save_experiment_results
from utils.init_utils import setup_gs_model
from utils.obstacle_utils import SCENARIOS, create_obstacle_scenario, obstacle_mask_to_geometry
from utils.sim_utils import (
    EnvironmentContext,
    GroundTruth,
    MeasurementBatch,
    generate_fractal_gas_distribution,
    generate_obstacle_aware_beams,
    simulate_gas_integrals,
)


PENALTY_WEIGHTS = (0.0, 0.1, 1.0)


def validate_grid(cfg: Config) -> tuple[int, int]:
    """Return grid dimensions and reject inconsistent physical dimensions."""
    map_w, map_h = cfg.env.map_size
    width_cells = map_w / cfg.env.cell_size
    height_cells = map_h / cfg.env.cell_size
    if not np.isclose(width_cells, round(width_cells)) or not np.isclose(height_cells, round(height_cells)):
        raise ValueError("map_size dimensions must be divisible by cell_size")
    return int(round(height_cells)), int(round(width_cells))


def create_toy_environment(
    cfg: Config,
    scenario: str,
) -> tuple[MeasurementBatch, EnvironmentContext]:
    """Create gas, obstacles, and measurements for one obstacle scenario."""
    grid_h, grid_w = validate_grid(cfg)
    obstacles = create_obstacle_scenario(scenario, (grid_h, grid_w))

    gas_gt = generate_fractal_gas_distribution(
        grid_size=(grid_h, grid_w), scale_fraction=0.3, center_bias=0.0
    )
    gas_gt[obstacles > 0.5] = 0.0

    obstacle_geometry = obstacle_mask_to_geometry(obstacles, cfg.env.cell_size)
    beams_list = generate_obstacle_aware_beams(
        cfg.env.map_size,
        cfg.sim.num_beams,
        obstacle_geometry=obstacle_geometry,
        seed=cfg.sim.seed,
    )
    measurements_list = simulate_gas_integrals(
        gas_gt, beams_list, cfg.env.cell_size, quiet=cfg.quiet
    )
    beams = torch.tensor(beams_list, dtype=torch.float32, device=cfg.device)
    y_true = torch.tensor(measurements_list, dtype=torch.float32, device=cfg.device)

    return (
        MeasurementBatch(beams=beams, measurements=y_true),
        EnvironmentContext(
            obstacles=obstacles,
            ground_truth=GroundTruth(gas_map=gas_gt, y_true=y_true),
        ),
    )


def calculate_metrics(
    final_map: np.ndarray,
    batch: MeasurementBatch,
    environment: EnvironmentContext,
    model: torch.nn.Module,
) -> dict[str, float]:
    """Calculate obstacle, reconstruction, and measurement metrics."""
    ground_truth = environment.ground_truth.gas_map
    occupied = environment.obstacles > 0.5
    free = ~occupied
    boundary = binary_dilation(occupied, iterations=1) & free
    free_area = max(int(free.sum()), 1)
    boundary_area = max(int(boundary.sum()), 1)

    with torch.no_grad():
        data_mae = torch.mean(torch.abs(model(batch.beams) - batch.measurements)).item()

    return {
        "obstacle_leakage": float(np.mean(final_map[occupied])) if occupied.any() else 0.0,
        "obstacle_max": float(np.max(final_map[occupied])) if occupied.any() else 0.0,
        "free_mae": float(np.sum(np.abs(final_map[free] - ground_truth[free])) / free_area),
        "boundary_mae": float(np.sum(np.abs(final_map[boundary] - ground_truth[boundary])) / boundary_area),
        "global_rmse": float(rmse_loss(ground_truth, final_map)),
        "data_mae": data_mae,
        "num_gaussians": float(model.num_gaussians),
    }


def run_test(
    scenario: str,
    penalty_weight: float,
    cfg: Config,
    data: tuple[MeasurementBatch, EnvironmentContext],
    seed: int | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train and evaluate one geometry/penalty combination."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    batch, environment = data
    test_cfg = copy.deepcopy(cfg)
    test_cfg.train.obstacle_lambda = penalty_weight

    start_time = time.time()
    model, _ = setup_gs_model(batch, test_cfg)
    setup_time = time.time() - start_time
    trainer = Trainer(model=model, cfg=test_cfg, environment=environment)
    batch_results = trainer.train(batch)
    training_results = trainer.finish()

    final_map = model.render_map(test_cfg.env.cell_size)
    metrics = calculate_metrics(final_map, batch, environment, model)
    metrics.update({
        "scenario": scenario,
        "obstacle_lambda": penalty_weight,
        "seed": seed,
        "setup_time": setup_time,
        "training_time": training_results.training_time,
        "final_loss": batch_results.final_loss,
    })
    print(
        f"{scenario:>14} | lambda={penalty_weight:<4g} | "
        f"leak={metrics['obstacle_leakage']:.4f} | "
        f"free MAE={metrics['free_mae']:.4f} | "
        f"RMSE={metrics['global_rmse']:.4f}"
    )
    return final_map, metrics


def plot_scenario(
    scenario: str,
    data: tuple[MeasurementBatch, EnvironmentContext],
    results: dict[float, tuple[np.ndarray, dict[str, float]]],
    cfg: Config,
    output_dir: str,
) -> None:
    """Save a comparable ground-truth and reconstruction figure."""
    batch, environment = data
    ground_truth = environment.ground_truth.gas_map
    maps = [ground_truth, *(item[0] for item in results.values())]
    vmax = max(float(map_result.max()) for map_result in maps)
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
    axes = np.atleast_1d(axes)
    red_cmap = mcolors.ListedColormap(["none", "red"])

    map_w, map_h = cfg.env.map_size
    extent = (0.0, map_w, 0.0, map_h)
    axes[0].imshow(ground_truth, origin="lower", vmin=0, vmax=vmax, cmap="viridis", extent=extent)
    axes[0].set_title(f"Ground truth\n{scenario}")
    for beam in batch.beams.detach().cpu().numpy():
        axes[0].plot(
            beam[:, 0],
            beam[:, 1],
            color="white", alpha=0.25, linewidth=0.5,
        )
    axes[0].imshow(environment.obstacles, cmap=red_cmap, origin="lower", alpha=0.45, extent=extent)

    for axis, (penalty_weight, (map_result, metrics)) in zip(axes[1:], results.items()):
        axis.imshow(map_result, origin="lower", vmin=0, vmax=vmax, cmap="viridis", extent=extent)
        axis.imshow(environment.obstacles, cmap=red_cmap, origin="lower", alpha=0.45, extent=extent)
        axis.set_title(
            f"lambda={penalty_weight:g}\n"
            f"leak={metrics['obstacle_leakage']:.3f}, "
            f"free MAE={metrics['free_mae']:.3f}"
        )

    for axis in axes:
        axis.set_xlim(0.0, map_w)
        axis.set_ylim(0.0, map_h)
        axis.axis("off")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"obstacles_{scenario}.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Compare obstacle geometries and penalties")
    parser.add_arguments(Config, dest="cfg")
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS),
        help="Obstacle scenarios to run",
    )
    parser.add_argument(
        "--penalty_weights", nargs="+", type=float, default=list(PENALTY_WEIGHTS),
        help="Obstacle penalty weights to compare",
    )
    args = parser.parse_args()
    cfg: Config = args.cfg
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    seed = cfg.sim.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    print(f"Using device: {cfg.device}")
    all_metrics = []
    for scenario in args.scenarios:
        print(f"\n--- Scenario: {scenario} ---")
        if seed is not None:
            np.random.seed(seed)
        data = create_toy_environment(cfg, scenario)
        scenario_results = {}
        for penalty_weight in args.penalty_weights:
            map_result, metrics = run_test(scenario, penalty_weight, cfg, data, seed)
            scenario_results[penalty_weight] = (map_result, metrics)
            all_metrics.append(metrics)
        plot_scenario(scenario, data, scenario_results, cfg, os.path.join(root_dir, "plots"))

    save_experiment_results(
        metadata={
            "experiment_name": "obstacle_geometry_comparison",
            "scenarios": args.scenarios,
            "penalty_weights": args.penalty_weights,
            "seed": seed,
            "map_size": cfg.env.map_size,
            "cell_size": cfg.env.cell_size,
        },
        results=all_metrics,
        folder=os.path.join(root_dir, "results"),
    )


if __name__ == "__main__":
    main()
