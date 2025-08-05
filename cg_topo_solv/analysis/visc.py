import numpy as np
from sklearn.linear_model import LinearRegression
from glob import glob
import os
import re

import networkx as nx
import random
from scipy.optimize import curve_fit


def get_meta_counts(data_dir):
    """Count metadata occurrences in rv log files under the given directory."""
    
    files = sorted(glob(os.path.join(data_dir, "rv", "rv_*9.9999*.log")))

    meta_counts = {}
    for file in files:
        if "EXTRA_HI" in file:
            meta = "_".join(file.split("rv_")[-1].split("EXTRA_HI_")[1].split("_9.9999")[0].split("_")[:-1])
            if len(meta.split("_")) == 8:
                meta_counts[meta] = meta_counts.get(meta, 0) + 1
        elif "EXTRA_MID" in file:
            meta = "_".join(file.split("rv_")[-1].split("EXTRA_MID_")[1].split("_9.9999")[0].split("_")[:-1])
            if len(meta.split("_")) == 8:
                meta_counts[meta] = meta_counts.get(meta, 0) + 1
        else:
            meta = "_".join(file.split("rv_")[-1].split("_meta")[0].split("_")[1:])
            meta_counts[meta] = meta_counts.get(meta, 0) + 1
        
    return meta_counts


def load_data(file_path, fallback_max_rows=20000, verbose=False):
    """Load numeric data from a text file, with error handling and fallback logic."""
    try:
        data = np.loadtxt(file_path, skiprows=1)
        return data if data.size > 0 else None
    except ValueError as e:
        error_message = str(e)
        if verbose:
            print(f"Initial load failed: {error_message}")
        
        match = re.search(r'at row (\d+)', error_message)
        if match:
            line_number = int(match.group(1))
            max_rows = line_number - 1
            if verbose:
                print(f"Retrying with max_rows = {max_rows}")
        else:
            if verbose:
                print("Could not find error row, using fallback max_rows.")
            max_rows = fallback_max_rows

        try:
            data = np.loadtxt(file_path, skiprows=1, max_rows=max_rows)
            return data if data.size > 0 else None
        except Exception as e2:
            if verbose:
                print(f"Retry failed: {e2}")
            return None
    except (IOError, Exception) as e:
        if verbose:
            print(f"Load failed: {e}")
        return None


def bootstrap_slope(x, y, num_samples=1000):
    """Estimate slope and its error using bootstrapped linear regressions."""
    x = np.array(x)
    y = np.array(y)
    slopes = []

    for _ in range(num_samples):
        idx = np.random.choice(len(x), len(x), replace=True)
        model = LinearRegression().fit(x[idx].reshape(-1, 1), y[idx])
        slopes.append(model.coef_[0])

    slopes = np.array(slopes)
    q1, q3 = np.percentile(slopes, [25, 75])
    iqr = q3 - q1
    mask = (slopes >= q1 - 1.5 * iqr) & (slopes <= q3 + 1.5 * iqr)
    clean_slopes = slopes[mask]

    return {
        "slope_mean": np.mean(clean_slopes),
        "slope_std": np.std(clean_slopes) / np.sqrt(len(clean_slopes)),
    }


def filter_linear_part(velocity, length, exclusion=1, hoomd_version="5.1.1"):
    """Extract and fit linear regions of velocity data for different HOOMD versions."""
    
    if hoomd_version == "2.9.5":
        x_left = velocity[50:147, 0].reshape(-1, 1)
        y_left = velocity[50:147, 1]
        
        x_right1 = velocity[:48, 0].reshape(-1, 1)
        y_right1 = velocity[:48, 1]
        
        x_right2 = velocity[148:, 0].reshape(-1, 1) - 120.0
        y_right2 = velocity[148:, 1] 
        x_right = np.concatenate((x_right1, x_right2), axis=0)
        y_right = np.concatenate((y_right1, y_right2)) * -1
        
    elif hoomd_version == "5.1.1":
        x_left  = velocity[1:99, 0].reshape(-1, 1)
        y_left  = velocity[1:99, 1]
        
        x_right = velocity[101:-1, 0].reshape(-1, 1)
        y_right = velocity[101:-1, 1] * -1
    
    model1 = LinearRegression().fit(x_left, y_left)
    r21 = model1.score(x_left, y_left)
    filtered1 = np.stack((x_left[..., 0], y_left), axis=1)
    
    model2 = LinearRegression().fit(x_right, y_right)
    r22 = model2.score(x_right, y_right)
    filtered2 = np.stack((x_right[..., 0], y_right), axis=1)
    
    return filtered1, filtered2, r21, r22



def calculate_shear_properties(momentum, velocity, samples, verbose=False, hoomd_version="5.1.1", momentum_freq=100):
    """Compute shear stress and velocity gradient statistics from momentum and velocity data."""
    
    if hoomd_version == "2.9.5":
        avg_momentum = np.mean(momentum[1:, 1])
        std_momentum = np.std(momentum[1:, 1]) / np.sqrt(len(momentum[1:, 1]))
        
    elif hoomd_version == "5.1.1":
        raw_momentum = momentum[1:]
        raw_momentum = np.diff(raw_momentum) / momentum_freq
        std_momentum = np.std(raw_momentum) / np.sqrt(len(raw_momentum))
        avg_momentum = np.mean(raw_momentum)
    
    velocity1, velocity2, r21, r22 = filter_linear_part(velocity, length=60, exclusion=0.5, hoomd_version=hoomd_version)
        
    if (r21 < 0.9 or r22 < 0.9) and verbose:
        print(f"Warning: Poor linear fit (R2 = {r21:.3f})")
        print(f"Warning: Poor linear fit (R2 = {r22:.3f})")

    summary1 = bootstrap_slope(velocity1[:, 0], velocity1[:, 1], samples)
    summary2 = bootstrap_slope(velocity2[:, 0], velocity2[:, 1], samples)
    
    return avg_momentum, std_momentum, summary1, summary2


def calc_viscosity(data_dir, meta, num_samples=100, size=120, valid_moments=[], verbose=False):
    """Compute viscosity from velocity and momentum data for a given metadata tag."""
    stresses, stresses_stds, rates, rate_stds = [], [], [], []
    
    velocity_paths = sorted(glob(os.path.join(data_dir, "vx", f"vx_{meta}.npy")))
    
    if len(velocity_paths) == 0:
        velocity_paths = sorted(glob(os.path.join(data_dir, "vl", f"vl_{meta}.npy")))
        print(os.path.join(data_dir, "vl", f"vl_{meta}.npy"))

    for velocity_path in velocity_paths:
        if "vx_" in velocity_path:
            meta_info = velocity_path.split("vx_")[-1].split(".npy")[0]
        elif "vl_" in velocity_path:
            meta_info = velocity_path.split("vl_")[-1].split(".npy")[0]
        
        current_moment = float(meta_info.split("_")[-3])
        current_marker = float(meta_info.split("_")[-2])
        
        if meta_info.split("_")[0] == "LOWDOP":
            current_marker = 9.9999

        momentum_path = os.path.join(data_dir, "rv", f"rv_{meta_info}.log")

        if current_moment not in valid_moments:
            continue
        
        if current_marker == 1.5000:
            hoomd_version = "2.9.5"
            momentum = load_data(momentum_path)
            velocity = load_data(velocity_path)
            
        elif current_marker == 9.9999:
            hoomd_version = "5.1.1"
            momentum = load_data(momentum_path)
            position = np.linspace(-59.7, 59.7, 200)
            
            velocity = np.load(velocity_path)
            velocity_mean = velocity[1:].mean(axis=0)
            velocity = np.stack((position, velocity_mean), axis=1)

        if momentum is None or velocity is None:
            continue

        avg_mmt, std_mmt, summary1, summary2 = calculate_shear_properties(momentum, velocity, num_samples, verbose=verbose, hoomd_version=hoomd_version)
        stress = avg_mmt / (2 * size**2 * 0.01)
        stress_std = std_mmt / (2 * size**2 * 0.01)

        stresses.append(stress)
        stresses_stds.append(stress_std)
        rates.append((summary1["slope_mean"] + summary2["slope_mean"]) / 2)
        rate_stds.append((summary1["slope_std"] + summary2["slope_std"]) / 2)

    if not stresses:
        return None

    stresses = np.array(stresses)
    stresses_stds = np.array(stresses_stds)
    rates = np.array(rates)
    rate_stds = np.array(rate_stds)

    viscosities = stresses / rates
    viscosity_stds = viscosities * np.sqrt((stresses_stds / stresses) ** 2 + (rate_stds / rates) ** 2)

    mean_visc = np.mean(viscosities)
    mask = np.abs(viscosities - mean_visc) <= 5 * np.std(viscosities)

    return {
        "shear_rate": np.column_stack((rates[mask], rate_stds[mask])),
        "mean_viscosity": viscosities[mask],
        "std_viscosity": viscosity_stds[mask],
        "stress": stresses[mask],
    }
    
    
def carreau_yasuda(shear_rate, eta_0, eta_inf, lambda_c, a, n):
    return eta_inf + (eta_0 - eta_inf) * (1 + (lambda_c * shear_rate)**a)**((n-1)/a)


def fit_carreau_yasuda_model(x_data, y_data, y_std=None):
    random.seed(0)
    np.random.seed(0)
    p0 = [np.max(y_data), np.min(y_data), 1e2, 2.0, 0.99]

    bounds = (
        [3.71, 3.71, 1e2, 0.1, 0.01],
        [100, 100, 1e5, 5.0, 0.99],
    )


    def model_with_constraint(x, eta_0, eta_inf, lambda_c, a, n):
        if eta_inf > eta_0:
            return np.full_like(x, 1e10)
        if abs(eta_0 - eta_inf) < 0.1:
            return np.full_like(x, 1e3) * (n-0.99) ** 2
        return carreau_yasuda(x, eta_0, eta_inf, lambda_c, a, n)

    try:
        popt, pcov = curve_fit(
            model_with_constraint,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            sigma=y_std,
            absolute_sigma=True,
            maxfev=20000,
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
        )
    except RuntimeError:
        p0 = [np.quantile(y_data, 0.9), np.quantile(y_data, 0.1), 1e2, 0.7, 0.5]
        popt, pcov = curve_fit(
            model_with_constraint,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            sigma=y_std,
            absolute_sigma=True,
            maxfev=20000,
            method="trf",
        )

    return popt, None
    
    
def process_viscosity_data(output):
    """
    Fit the Carreau-Yasuda model.
    """
    result_new = dict()
    result_new["shear_rate"] = output["shear_rate"][:, 0]
    result_new["mean_viscosity"] = output["mean_viscosity"]
    result_new["std_viscosity"] = output["std_viscosity"]

    xx_raw = result_new["shear_rate"]
    yy_raw = result_new["mean_viscosity"]
    yy_std = result_new["std_viscosity"]
    
    p, result = fit_carreau_yasuda_model(xx_raw, yy_raw, yy_std)

    new_xx = np.logspace(-4, np.log10(0.03), 100)
    
    new_yy = carreau_yasuda(new_xx, *p)
    
    return result_new, new_xx, new_yy, p