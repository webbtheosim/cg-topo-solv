from cg_topo_solv.ml.vae import get_file_names, train_vae
from cg_topo_solv.ml.data import load_data

import numpy as np
import os
import itertools
from tensorflow.keras import backend as K

################ USER CAN ADJUST DIR ################
from pathlib import Path
BASE_DIR = Path("/scratch/gpfs/sj0161/")
WEIGHT_DIR = BASE_DIR / "mpcd_ml_weight"
ANALYSIS_DIR = BASE_DIR / "mpcd_ml_analysis"
DATA_FILE = BASE_DIR / "data_ml" / "data_aug10.pickle"
################ USER CAN ADJUST DIR ################


class Args:
    def __init__(self):
        self.lr = 1e-3
        self.bs = 32
        self.rw = 10.0
        self.cw = 10.0
        self.max_lambda = 0.5


def load_ml_data(max_lambda=0.5, if_test=False, if_verbose=False):
    """Load data for ML model"""
    (
        (x_train, y_train, c_train, l_train, graph_train),
        (x_valid, y_valid, c_valid, l_valid, graph_valid),
        (x_test, y_test, c_test, l_test, graph_test),
        NAMES,
        SCALER,
        SCALER_y,
        LE,
    ) = load_data(
        data_dir=DATA_FILE,
        fold=0,
        if_validation=True,
        rerun=False,
        max_lambda=max_lambda,
    )

    has_nan = np.isnan(l_train).any()
    has_inf = np.isinf(l_train).any()

    if if_verbose:
        print("l_train contains NaN values:", has_nan)
        print("l_train contains inf values:", has_inf)
        print(f"l_train shape: {l_train.shape}")

    if if_test:
        return (
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
            LE
        )
    else:
        return x_train, x_valid, y_train, y_valid, c_train, c_valid, l_train, l_valid


def train_ml(args):
    """Train VAE model"""
    x_train, x_valid, y_train, y_valid, c_train, c_valid, l_train, l_valid = (
        load_ml_data(max_lambda=args.max_lambda)
    )

    K.clear_session()

    LATENT_DIM = 8
    DECODER = "cnn"
    ENCODER = "desc_gnn"
    MONITOR = "val_decoder_acc"
    IF_REG = True
    IF_CLS = True

    LR = args.lr
    BS = args.bs
    rw = args.rw
    cw = args.cw

    weight_name, hist_name = get_file_names(
        ENCODER,
        DECODER,
        "20240816",
        LATENT_DIM,
        MONITOR,
        IF_REG,
        IF_CLS,
        1.0,
        [1.0, rw, cw],
        LR,
        BS,
    )

    if os.path.exists(os.path.join(WEIGHT_DIR, hist_name)):
        if_train = False
    else:
        if_train = True

    print(f"Training started: {weight_name}")

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
        [1.0, rw, cw],
        LR,
        BS,
        if_train,
        verbose=2,
        n_epoch=1000,
        date="20240816",
    )

    print(f"Training ended: {weight_name}")


def job_array(idx, max_idx):
    """Generate hyperparameters for job array"""
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    bss = [32, 64, 128, 256]
    rws = [0.01, 0.1, 1.0, 10.0, 100.0]
    cws = [0.01, 0.1, 1.0, 10.0, 100.0]

    combs = itertools.product(lrs, bss, rws, cws)
    combs = list(combs)

    size = len(combs) // max_idx
    start = idx * size
    end = min((idx + 1) * size, len(combs))

    return combs[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1
    combs = job_array(idx, max_idx)

    for comb in combs:
        lr, bs, rw, cw = comb
        args = Args()
        args.lr = lr
        args.bs = bs
        args.rw = rw
        args.cw = cw

        train_ml(args)
