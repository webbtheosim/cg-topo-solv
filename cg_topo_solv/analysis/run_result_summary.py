import time
from cg_topo_solv.analysis.result import load_target, load_all_fit


def run_sum_result(result_path, simulation_path, rerun=True):
    """Summarize simulation results by loading target curves and fitted values."""
    start = time.time()

    idx_hi, idx_mid, idx_lo, curves, samples_y, params = load_target(result_path, verbose=True)
    all_fit = load_all_fit(result_path, simulation_path, rerun=rerun)

    print(f"Total runtime: {time.time() - start:.2f} s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize MPCD results.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--simulation_path", type=str, required=True, help="Path to the simulation directory.")
    parser.add_argument("--rerun", action="store_true", help="Rerun the analysis even if results exist.")

    args = parser.parse_args()

    run_sum_result(args.result_path, args.simulation_path, rerun=args.rerun)