import os
import glob
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from tensorflow.keras import backend as K

from cg_topo_solv.ml.train import train_vae, load_ml_data
from cg_topo_solv.ml.vae import get_spec, latent_model

################ USER CAN ADJUST DIR ################
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHT_DIR = BASE_DIR / "mpcd_ml_weight"
ANALYSIS_DIR = BASE_DIR / "mpcd_ml_analysis"
DATA_FILE = BASE_DIR / "data_ml" / "data_aug20.pickle"
VISCO_DATA_DIR = BASE_DIR / "mpcd" / "result_1106"
################ USER CAN ADJUST DIR ################


class Args:
    def __init__(self):
        self.lr = 1e-3
        self.bs = 32
        self.rw = 10.0
        self.cw = 10.0
        self.max_lambda = 0.5
        self.VISCO_DATA_DIR = VISCO_DATA_DIR
        self.WEIGHT_DIR = ANALYSIS_DIR
        self.ANALYSIS_DIR = ANALYSIS_DIR
        self.DATA_FILE = DATA_FILE


def glob_weight(weight_dir):
    """Glob weight files"""
    weight_files = glob.glob(f"{weight_dir}*.h5")
    weight_files = sorted(weight_files)
    return weight_files


def cv_select():
    """
    Perform model selection based on combined score and save the results to a CSV file.
    Returns:
        np.ndarray: A numpy array containing the computed metrics for each model.
    Notes:
        - If the output CSV file already exists, the function reads the metrics from the file.
        - If the output CSV file does not exist, the function loads the machine learning data,
          iterates over weight files, trains the models, computes the metrics, and saves them
          to the CSV file.
    Metrics:
        - KL Divergence: Measures the divergence between the latent variable distributions.
        - Balanced Accuracy: Measures the balanced accuracy score of the predictions.
        - R2 Score: Measures the coefficient of determination for regression predictions.
        - F1 Score: Measures the F1 score for classification predictions.
    """

    args = Args()
    output_file = os.path.join(args.ANALYSIS_DIR, "cv_select.csv")

    if os.path.exists(output_file):
        metrics_df = pd.read_csv(output_file)
        metrics = metrics_df.values
        weight_files = glob_weight(args.WEIGHT_DIR)
    else:  
        x_train, x_valid, y_train, y_valid, c_train, c_valid, l_train, l_valid = load_ml_data(max_lambda=args.max_lambda)
        weight_files = glob_weight(args.WEIGHT_DIR)
        metrics = []

        for weight_file in weight_files:
            ENCODER, DECODER, MONITOR, IF_REG, IF_CLS, weights, LR, BS = get_spec(weight_file)
            K.clear_session()

            model, pickle_file = train_vae(
                ENCODER, DECODER, MONITOR, IF_REG, IF_CLS,
                x_train, x_valid, y_train, y_valid,
                c_train, c_valid, l_train, l_valid,
                1.0, weights, LR, BS,
                if_train=False, verbose=2, n_epoch=1000, date="20240816"
            )

            if ENCODER == "desc_dnn":
                in_valid = np.copy(l_valid)
            elif ENCODER == "desc_gnn":
                in_valid = [[np.copy(x_valid), np.copy(x_valid)], np.copy(l_valid)]
            elif ENCODER == "gnn":
                in_valid = [np.copy(x_valid), np.copy(x_valid)]
            else:
                in_valid = np.copy(x_valid)

            y_pred = model.predict(in_valid)

            latent_valid = latent_model(
                model, data=[np.copy(x_valid), np.copy(l_valid)],
                enc_type=ENCODER, mean_var=True
            )

            z_mean = latent_valid[0]
            z_log_var = latent_valid[1]

            kl_total = -0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=-1) * 1
            kl = kl_total.mean()

            bacc = skm.balanced_accuracy_score(x_valid.ravel(), np.round(y_pred[0]).ravel())
            r2 = skm.r2_score(y_valid.ravel(), y_pred[1].ravel())
            f1 = skm.f1_score(c_valid, np.argmax(y_pred[2], axis=1).ravel(), average="micro")

            K.clear_session()
            metrics.append([kl, bacc, r2, f1])

            metrics_df = pd.DataFrame(metrics, columns=["KL Divergence", "Balanced Accuracy", "R2 Score", "F1 Score"])
            metrics_df.to_csv(output_file, index=False)

    return np.array(metrics), weight_files


def minmax(arr, mins=None, maxs=None):
    """
    Perform min-max scaling on an array.

    Args:
        arr (numpy.ndarray): The input array to be scaled.
        mins (float, optional): The custom minimum value for scaling. Default is None.
        maxs (float, optional): The custom maximum value for scaling. Default is None.

    Returns:
        numpy.ndarray: The scaled array.
    """
    # Determine minimum and maximum values for scaling
    if mins is not None:
        min_val = mins
    else:
        min_val = np.min(arr)

    if maxs is not None:
        max_val = maxs
    else:
        max_val = np.max(arr)

    # Perform min-max scaling
    scaled_arr = [(x - min_val) / (max_val - min_val) for x in arr]

    return scaled_arr


def pareto_frontier(baccs, r2s, f1s, kls, limits):
    """
    Find the Pareto frontier from a set of data points with multiple objectives.

    Args:
        baccs (list): List of balanced accuracy values.
        r2s (list): List of R2 values.
        f1s (list): List of F1 scores.
        kls (list): List of KL divergence values.
        limits (list): List of limit values for each objective.

    Returns:
        tuple: A tuple containing two elements:
            - pareto_indices (list): List of indices of Pareto points.
            - pareto_front (numpy.ndarray): Array of Pareto points satisfying the specified limits.
    """
    combined = list(zip(baccs, r2s, f1s, kls, range(len(baccs))))
    pareto_front = []

    for point in combined:
        is_dominated = False

        for other_point in combined:
            if all(
                other <= point_dim
                for other, point_dim in zip(other_point[:4], point[:4])
            ) and any(
                other < point_dim
                for other, point_dim in zip(other_point[:4], point[:4])
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(point)

    pareto_indices = [
        point[4]
        for point in pareto_front
        if all(point_dim < limit for point_dim, limit in zip(point[:4], limits))
    ]

    pareto_front = [
        point[:5]
        for point in pareto_front
        if all(point_dim < limit for point_dim, limit in zip(point[:4], limits))
    ]

    return pareto_indices, np.array(pareto_front)


def closest_to_origin(pareto_front):
    """
    Find the point in the Pareto front closest to the origin in multi-dimensional space.

    Args:
        pareto_front (list): List of points in the Pareto front, each represented as a tuple of objectives.

    Returns:
        tuple: A tuple containing two elements:
            - closest_point (tuple): The point in the Pareto front closest to the origin.
            - closest_idx (int): The index of the closest point in the Pareto front.
    """
    baccs, r2s, f1s, kls, idx = zip(*pareto_front)

    baccs = minmax(baccs)
    r2s = minmax(r2s)
    f1s = minmax(f1s)
    kls = minmax(kls)

    scaled_pareto_front = list(zip(baccs, r2s, f1s, kls, idx))
    closest_tuple = min(
        scaled_pareto_front, key=lambda point: np.linalg.norm(np.array(point[:-1]))
    )

    closest_point = closest_tuple[:-1]
    closest_idx = closest_tuple[-1]

    return closest_point, int(closest_idx)

