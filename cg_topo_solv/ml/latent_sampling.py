import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
from sklearn.utils import resample
from tensorflow.keras import backend as K

import kmedoids

from cg_topo_solv.ml.cv_select import closest_to_origin, cv_select, pareto_frontier
from cg_topo_solv.ml.train import load_ml_data, train_vae
from cg_topo_solv.ml.vae import get_spec, latent_model


class Args:
    def __init__(self):
        self.lr = 1e-3
        self.bs = 32
        self.rw = 10.0
        self.cw = 10.0
        self.max_lambda = 0.5
        self.DATA_DIR = "/scratch/gpfs/sj0161/topo_data/"
        self.VISCO_DATA_DIR = "/scratch/gpfs/sj0161/mpcd/result_1106/"
        self.WEIGHT_DIR = "/scratch/gpfs/sj0161/mpcd_ml_weight/"
        self.ANALYSIS_DIR = "/scratch/gpfs/sj0161/mpcd_ml_analysis/"
        self.DATA_FILE = "/scratch/gpfs/sj0161/mpcd/data_ml/data_aug20.pickle"


def maximin(domain, size, seed):
    """
    Select a sample that maximizes the minimum distance
    between any two points in the sample, subject to
    a randomly chose initial point.
    """
    np.random.seed(seed=seed)
    domain = MinMaxScaler().fit_transform(domain)

    sample = [np.random.randint(low=0, high=domain.shape[0])]

    for _ in range(size - 1):
        chosen_points = domain[sample, :]
        distances = cdist(chosen_points, domain)
        distances = np.array(distances)
        distances = np.min(distances, axis=0).reshape(-1, 1)

        new_id = np.argsort(-distances, axis=0)[0].item()

        sample.append(new_id)

    return sample


def medoids(domain, size, seed):
    """
    Select a sample that minimizes the variance between
    all points in the dataset to a selected point.
    """

    # Scale data.
    domain = MinMaxScaler().fit_transform(domain)

    # Fit k-medoids.
    k = kmedoids.KMedoids(n_clusters=size, metric="euclidean", random_state=seed)
    output = k.fit(X=domain, y=None)
    sample = output.medoid_indices_.tolist()

    return sample


def medoids_bin(domain, y, size, seed, nbin=6):
    """
    Select a sample that minimizes the variance between
    all points in the dataset to a selected point, considering bins of y.
    """

    domain = StandardScaler().fit_transform(domain)

    binner = KBinsDiscretizer(n_bins=nbin, encode="onehot", strategy="uniform")
    y_binned = binner.fit_transform(y.reshape(-1, 1)).flatten()

    combined_sample_indices = []

    for bin_value in np.unique(y_binned):
        bin_indices = np.where(y_binned == bin_value)[0]
        bin_domain = domain[bin_indices]

        if len(bin_domain) < size:
            continue

        k = kmedoids.KMedoids(n_clusters=size, metric="euclidean", random_state=seed)
        k.fit(X=bin_domain, y=None)

        medoid_indices_within_bin = k.medoid_indices_
        medoid_indices_original = bin_indices[medoid_indices_within_bin]

        combined_sample_indices.extend(medoid_indices_original)

    if len(combined_sample_indices) < size:
        remaining_indices = list(set(range(len(y))) - set(combined_sample_indices))
        additional_sample_indices = resample(
            remaining_indices,
            n_samples=size - len(combined_sample_indices),
            random_state=seed,
        )
        combined_sample_indices.extend(additional_sample_indices)

    return combined_sample_indices


def medoids_sampling(domain, y, size, seed):
    """
    Perform medoid sampling from the dataset while maintaining the proportions of each class in y.

    Parameters:
    - domain: Features of the data (numpy array or similar).
    - y: Class labels corresponding to the domain (1D numpy array).
    - size: Total number of samples desired.
    - seed: Random seed for reproducibility.
    - nbin: Number of unique class labels (default: 6).

    Returns:
    - combined_sample_indices: List of indices representing the selected sample.
    """
    np.random.seed(seed)
    domain = StandardScaler().fit_transform(domain)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_ratios = class_counts / len(y)

    combined_sample_indices = []

    for class_value, class_ratio in zip(unique_classes, class_ratios):
        n_samples = max(1, int(size * class_ratio))

        class_indices = np.where(y == class_value)[0]
        class_domain = domain[class_indices]

        if len(class_domain) <= n_samples:
            combined_sample_indices.extend(class_indices)
        else:
            k = kmedoids.KMedoids(
                n_clusters=n_samples, metric="euclidean", random_state=seed
            )
            k.fit(X=class_domain)

            medoid_indices_within_class = k.medoid_indices_
            medoid_indices_original = class_indices[medoid_indices_within_class]

            combined_sample_indices.extend(medoid_indices_original)
            
    if len(combined_sample_indices) < size:
        remaining_indices = list(set(range(len(y))) - set(combined_sample_indices))
        additional_sample_indices = resample(
            remaining_indices,
            n_samples=size - len(combined_sample_indices),
            random_state=seed,
        )
        combined_sample_indices.extend(additional_sample_indices)

    return combined_sample_indices


def maximin_sampling(domain, y, size, seed):
    """
    Perform maximin sampling from the dataset while maintaining the proportions of each class in y.

    Parameters:
    - domain: Features of the data (numpy array or similar).
    - y: Class labels corresponding to the domain (1D numpy array).
    - size: Total number of samples desired.
    - seed: Random seed for reproducibility.
    - nbin: Number of unique class labels (default: 6).

    Returns:
    - combined_sample_indices: List of indices representing the selected sample.
    """
    np.random.seed(seed)
    domain = MinMaxScaler().fit_transform(domain)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_ratios = 1 / np.ones(len(unique_classes)) / len(unique_classes) #class_counts / len(y)

    combined_sample_indices = []

    for class_value, class_ratio in zip(unique_classes, class_ratios):
        n_samples = max(1, int(size * class_ratio))

        class_indices = np.where(y == class_value)[0]
        class_domain = domain[class_indices]

        if len(class_domain) <= n_samples:
            combined_sample_indices.extend(class_indices)
        else:
            sample = [np.random.choice(class_indices)]

            for _ in range(n_samples - 1):
                chosen_points = domain[sample, :]
                distances = cdist(chosen_points, class_domain)
                min_distances = np.min(distances, axis=0).reshape(-1, 1)

                # Sort points by minimum distance and find the point with the maximum minimum distance
                new_id = np.argsort(-min_distances, axis=0)[0].item()
                sample.append(class_indices[new_id])

            combined_sample_indices.extend(sample)
            
        print(f"Class {class_value}: {len(sample)} samples selected")

    return combined_sample_indices


def latent_sampling(seed=0, mode="maximin", rerun=False, size=96, verbose=False):
    args = Args()

    if size == 30:
        sample_file = os.path.join(args.ANALYSIS_DIR, f"sample_z_mean_0930_{mode}.pickle")
    else:
        sample_file = os.path.join(args.ANALYSIS_DIR, f"sample_z_mean_0930_{mode}_{size}.pickle")
        

    (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        c_train,
        c_valid,
        c_test,
        l_train,
        l_valid,
        l_test,
        graph_train,
        graph_valid,
        graph_test,
        SCALER,
        SCALER_y,
        LE,
    ) = load_ml_data(max_lambda=args.max_lambda, if_test=True)

    # Print dataset sizes and ratios
    total_samples = len(x_train) + len(x_valid) + len(x_test)
    train_ratio = len(x_train) / total_samples * 100
    valid_ratio = len(x_valid) / total_samples * 100
    test_ratio = len(x_test) / total_samples * 100
    
    if verbose:
        print(f"Training set:   {len(x_train)} samples ({train_ratio:.1f}%)")
        print(f"Validation set: {len(x_valid)} samples ({valid_ratio:.1f}%)")
        print(f"Test set:       {len(x_test)} samples ({test_ratio:.1f}%)")

    graph_all = np.concatenate((graph_train, graph_valid, graph_test), axis=0)
    x_all = np.concatenate((x_train, x_valid, x_test), axis=0)
    l_all = np.concatenate((l_train, l_valid, l_test), axis=0)
    y_all = np.concatenate((y_train, y_valid, y_test), axis=0)
    c_all = np.concatenate((c_train, c_valid, c_test), axis=0)

    if os.path.exists(sample_file) and not rerun:
        with open(sample_file, "rb") as f:
            data = pickle.load(f)
            sample = data["sample"]
            z_mean = data["z_mean"]
        print(f"Loaded sampled latent space points from sample_z_mean_{mode}.pickle")
    else:
        metrics, weight_files = cv_select()

        baccs = metrics[:, 1]
        r2s = metrics[:, 2]
        f1s = metrics[:, 3]
        kls = metrics[:, 0]

        pareto_indices, pareto_front = pareto_frontier(
            1 - baccs, 1 - r2s, 1 - f1s, kls, limits=[0.1, 0.1, 0.1, np.inf]
        )

        best_point, best_idx = closest_to_origin(pareto_front)

        best_weight = weight_files[best_idx]

        ENCODER, DECODER, MONITOR, IF_REG, IF_CLS, weights, LR, BS = get_spec(
            best_weight
        )

        K.clear_session()

        model, pickle_file = train_vae(
            ENCODER,
            DECODER,
            MONITOR,
            IF_REG,
            IF_CLS,
            x_train,
            x_valid,
            y_train,
            y_valid,
            c_train,
            c_valid,
            l_train,
            l_valid,
            1.0,
            weights,
            LR,
            BS,
            if_train=False,
            verbose=2,
            n_epoch=1000,
            date="20240816",
        )

        latent_valid = latent_model(
            model,
            data=[np.copy(x_all), np.copy(l_all)],
            enc_type=ENCODER,
            mean_var=True,
        )

        z_mean = latent_valid[0]

        if mode == "maximin":
            sample = maximin(z_mean, size, seed)
        elif mode == "medoids":
            sample = medoids(z_mean, size, seed)
        elif mode == "medoids_bin":
            sample = medoids_bin(z_mean, c_all, size, seed)
        elif mode == "medoids_sampling":
            sample = medoids_sampling(z_mean, c_all, size, seed)
        elif mode == "maximin_sampling":
            sample = maximin_sampling(z_mean, c_all, size, seed)
        elif mode == "maximin30_then_maximin":
            sample_first30 = maximin_sampling(z_mean, c_all, 30, seed)
            remaining_pool = np.delete(np.arange(z_mean.shape[0]), sample_first30)
            z_remaining = z_mean[remaining_pool]
            sample_rest = maximin(z_remaining, size - 30, seed + 1)
            sample_rest = remaining_pool[sample_rest]
            sample = sample_first30 + sample_rest.tolist()

        with open(sample_file, "wb") as f:
            pickle.dump({"sample": sample, "z_mean": z_mean}, f)

        print(f"Sampled latent space points saved to sample_z_mean_0930_{mode}.pickle")

    l_all = SCALER.inverse_transform(l_all)
    y_all = SCALER_y.inverse_transform(y_all)
    c_all = LE.inverse_transform(c_all)

    return sample, z_mean, y_all, c_all, graph_all
