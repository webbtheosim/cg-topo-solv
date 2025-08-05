import time
import os
import numpy as np
from itertools import product
from cg_topo_solv.analysis.result import result_visc


def run_sim_to_visc(result_path, simulation_path, rerun=False):
    """Process simulation outputs to compute and fit viscosity curves."""
    start = time.time()

    moments = np.array([
        0.1, 0.08, 0.06, 0.04, 0.030, 0.024, 0.018,
        0.012, 0.006, 0.003, 0.001, 0.0008, 0.0006, 0.0004
    ])

    batches = [1, 2, 3, 4, 5]
    methods = ["al", "csf", "sf"]
    
    result_visc(
        input_dir=simulation_path,
        output_dir=result_path,
        raw_visc_file="raw_visc_seed.pickle",
        fit_visc_file="fit_visc_seed.pickle",
        moments=moments,
        rerun_raw=rerun
    )

    for batch, method in product(batches, methods):
        subdir = f"batch_{batch}{method}"
        input_dir = os.path.join(simulation_path, subdir)

        raw_file = f"raw_visc_{method}_{batch}.pickle"
        fit_file = f"fit_visc_{method}_{batch}.pickle"

        result_visc(
            input_dir=input_dir,
            output_dir=result_path,
            raw_visc_file=raw_file,
            fit_visc_file=fit_file,
            moments=moments,
            rerun_raw=rerun
        )

    print(f"Total runtime: {time.time() - start:.2f} s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run viscosity analysis on simulation batches.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--simulation_path", type=str, required=True, help="Path to the simulation directory.")
    parser.add_argument("--rerun", action="store_true", help="Rerun the analysis even if results exist.")
    
    args = parser.parse_args()

    run_sim_to_visc(args.result_path, args.simulation_path, rerun=args.rerun)