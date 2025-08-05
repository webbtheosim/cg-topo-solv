import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from cg_topo_solv.analysis.visc import calc_viscosity, get_meta_counts
from cg_topo_solv.analysis.visc import process_viscosity_data
from cg_topo_solv.analysis.graph import get_desc
from cg_topo_solv.ml.latent_sampling import latent_sampling


def load_target(result_dir, verbose=False):
    """
    Load reference viscosity curves and classify them by extrapolation distance.
    Args:
        result_dir (str): Path to the folder containing pickle files.
        verbose (bool): Whether to print diagnostic info.
    """
    with open(os.path.join(result_dir, "target_curves.pickle"), 'rb') as f:
        data = pickle.load(f)
        curves = data['curves']
        params = data['params']
    
    # seed dataset    
    with open(os.path.join(result_dir, "fit_visc_seed.pickle"), "rb") as f:
        samples_x = pickle.load(f)
        samples_y = pickle.load(f)

    if verbose:
        print(f"Shapes: curves {curves.shape}, samples {np.array(samples_y).shape}")

    def _classify(curves, samples):
        d = cdist(curves, samples, metric='euclidean')
        d_top5 = np.sort(d, axis=1)[:, :5]
        d_avg = np.mean(d_top5, axis=1)
        sorted_idx = np.argsort(d_avg)[::-1]
        n = len(curves) // 3
        groups = {
            'Large': sorted_idx[:n],
            'Medium': sorted_idx[n:2*n],
            'Small': sorted_idx[2*n:]
        }
        if verbose:
            for name, idx in groups.items():
                stats = d_avg[idx]
                print(f"{name} distance stats:")
                print(f"  Mean: {np.mean(stats):.4f}")
                print(f"  Min:  {np.min(stats):.4f}")
                print(f"  Max:  {np.max(stats):.4f}")
                print(f"  STD:  {np.std(stats):.4f}")
        return groups['Large'], groups['Medium'], groups['Small']

    idx_hi, idx_mid, idx_lo = _classify(curves, np.array(samples_y))
    return idx_hi, idx_mid, idx_lo, curves, samples_y, params


def load_all_fit(result_path, rerun=True):
    """
    Load all fitted viscosity curves and parameters from the result directory.
    """
    
    output_path = os.path.join(result_path, "topo_param_visc.pickle")
    if os.path.exists(output_path) and not rerun:
        with open(output_path, "rb") as f:
            return pickle.load(f)

    def _load_batch_data(param_paths, sample_paths):
        desc_list, ps_list, curve_list, graph_list = [], [], [], []
        for param_path, sample_path in zip(param_paths, sample_paths):
            with open(sample_path, "rb") as f:
                graphs = pickle.load(f)
                props = np.array(pickle.load(f))
            desc = np.array([get_desc(g) for g in graphs])
            solv = props[:, -1].reshape(-1, 1)
            desc = np.concatenate((desc, solv), axis=1)
            desc = np.nan_to_num(desc, nan=-2)
            with open(param_path, "rb") as f:
                shear_rate, vis_curve, _, ps = (np.array(pickle.load(f)) for _ in range(4))
            desc_list.append(desc)
            ps_list.append(ps)
            curve_list.append(vis_curve)
            graph_list.extend(graphs)
        return (
            np.concatenate(desc_list),
            np.concatenate(ps_list),
            np.concatenate(curve_list),
            shear_rate,
            graph_list,
        )

    sample, z_mean, y_all, c_all, graph_all = latent_sampling(
        seed=42, mode="maximin_sampling", rerun=False, size=30
    )
    desc_seed = np.array([get_desc(graph_all[idx]) for idx in sample])
    solv_seed = y_all[sample][:, -1].reshape(-1, 1)
    desc_seed = np.concatenate((desc_seed, solv_seed), axis=1)
    desc_seed = np.where(np.isnan(desc_seed), -2, desc_seed)
    graph_seed = [graph_all[i] for i in sample]

    al_paths = [os.path.join(result_path, f"fit_visc_al_{i+1}.pickle") for i in range(5)]
    csf_paths = [os.path.join(result_path, f"fit_visc_csf_{i+1}.pickle") for i in range(5)]
    sf_paths = [os.path.join(result_path, f"fit_visc_sf_{i+1}.pickle") for i in range(5)]

    al_samples = [os.path.join(result_path, f"batch_{i+1}_sample_al.pickle") for i in range(5)]
    csf_samples = [os.path.join(result_path, f"batch{i+1}_sample_csf.pickle") for i in range(5)]
    sf_samples = [os.path.join(result_path, f"batch{i+1}_sample_sf.pickle") for i in range(5)]

    with open(os.path.join(result_path, "fit_visc_seed.pickle"), "rb") as f:
        xx_seed, yy_seed, topo_seed, ps_seed = (np.array(pickle.load(f)) for _ in range(4))

    desc_al, ps_al, curve_al, shear_rate, graph_al = _load_batch_data(al_paths, al_samples)
    desc_csf, ps_csf, curve_csf, _, graph_csf = _load_batch_data(csf_paths, csf_samples)
    desc_sf, ps_sf, curve_sf, _, graph_sf = _load_batch_data(sf_paths, sf_samples)

    desc_all = np.concatenate((desc_seed, desc_csf, desc_sf, desc_al))
    ps_all = np.concatenate((ps_seed, ps_csf, ps_sf, ps_al))
    curve_all = np.concatenate((yy_seed, curve_csf, curve_sf, curve_al))
    graph_all = graph_seed + graph_csf + graph_sf + graph_al

    with open(output_path, "wb") as f:
        pickle.dump((desc_all, ps_all, curve_all, shear_rate, graph_all), f)

    return desc_all, ps_all, curve_all, shear_rate, graph_all


def result_visc(input_dir="/scratch/gpfs/sj0161/mpcd/result_1106/",
                output_dir="/home/sj0161/mpcd_result/",
                raw_visc_file="raw_visc_seed.pickle",
                fit_visc_file="fit_visc_seed.pickle",
                rerun_raw=False,
                moments=[]):
    """
    Calculate and save viscosity data from simulation results.
    """

    meta_counts = get_meta_counts(data_dir=input_dir)
    meta_keys = list(meta_counts.keys())
    
    if meta_keys[0].split("_")[1] != "meta":
        meta_numbers = np.array([int(meta_key.split("_")[1]) for meta_key in meta_keys])
        meta_keys_new = []
        

        if "seed" in raw_visc_file:
            sample, z_mean, y_all, c_all, graph_all = latent_sampling(
            seed=42, mode="maximin_sampling", rerun=False, size=30
            )
            for i in range(30):
                idx = np.where(meta_numbers == i)[0][0]
                meta_keys_new.append(meta_keys[idx])
        else:
            sample = np.arange(30)

            for i in sample:
                idx = np.where(meta_numbers == i)[0][0]
                meta_keys_new.append(meta_keys[idx])
        

        meta_keys = meta_keys_new

    if rerun_raw:
        outputs = []
        print(meta_counts)
        
        for key in tqdm(meta_keys, total=len(meta_keys)):
            out = calc_viscosity(data_dir=input_dir,
                                meta="*" + key + "*",
                                num_samples=100,
                                size=120,
                                valid_moments=moments)
            outputs.append(out)

        raw_path = os.path.join(output_dir, raw_visc_file)
        with open(raw_path, "wb") as f:
            pickle.dump(outputs, f)
    else:
        with open(os.path.join(output_dir, raw_visc_file), "rb") as f:
            outputs = pickle.load(f)

    fit_x, fit_y, fit_meta, fit_params = [], [], [], []
    for i, out in enumerate(outputs):
        
        _, xs, ys, params = process_viscosity_data(out)
        fit_x.append(xs)
        fit_y.append(ys)
        fit_meta.append(meta_keys[i])
        fit_params.append(params)

        param_str = "  ".join(f"{p:>10.2f}" for p in params)
        print(f"{meta_keys[i]:<24}:  {param_str}")
        
    fit_path = os.path.join(output_dir, fit_visc_file)
    
    with open(fit_path, "wb") as f:
        pickle.dump(fit_x, f)
        pickle.dump(fit_y, f)
        pickle.dump(fit_meta, f)
        pickle.dump(fit_params, f)