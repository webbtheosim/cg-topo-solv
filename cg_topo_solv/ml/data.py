import pickle
import numpy as np
import os
import scipy.integrate as integrate
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

################ USER CAN ADJUST DIR ################
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHT_DIR = BASE_DIR / "mpcd_ml_weight"
ANALYSIS_DIR = BASE_DIR / "mpcd_ml_analysis"
DATA_FILE = BASE_DIR / "data_ml" / "data_aug10.pickle"
VISCO_DATA_DIR = BASE_DIR / "mpcd" / "result_1106"
TOPORG1342_FILE = BASE_DIR / "topo_data" / "rg2.pickle"
################ USER CAN ADJUST DIR ################


def ashbaugh(r, epsilon, sigma, lam):
    """ Ashbaugh potential """
    xi = sigma / r

    if r <= 2 ** (1 / 6):
        return 4 * epsilon * (xi**12 - xi**6) + (1 - lam) * epsilon
    else:
        return 4 * epsilon * (xi**12 - xi**6) * lam


def ashbaugh_integrand(r, epsilon, sigma, beta, lam):
    """ Integrand of Ashbaugh potential """
    return (1- np.exp(-beta * ashbaugh(r, epsilon, sigma, lam))) * r**2


def calculate_b2(lam):
    """ Calculate B2 for a given lambda """
    kbt = 1.0
    beta = 1.0 / kbt
    epsilon = 1.0
    sigma = 1.0
    result, error = integrate.quad(
        ashbaugh_integrand, 0, np.inf, args=(epsilon, sigma, beta, lam)
    )
    B2 = 2 * np.pi * result
    return B2


def extract_descriptors(graph_list, lam):
    """ Extract chain virial coefficient """
    descriptors = []
    for g, lam_temp in zip(graph_list, lam):
        num_nodes = g.number_of_nodes()
        Xi_temp = calculate_b2(lam_temp) * num_nodes
        descriptors.append(Xi_temp)
    return np.array(descriptors)


def load_data(
    data_dir,
    fold,
    n_fold=5,
    n_repeat=5,
    if_validation=True,
    verbose=False,
    rerun=False,
    max_lambda=0.5,
):

    if os.path.exists(data_dir) and not rerun:
        with open(data_dir, "rb") as handle:
            (
                (x_train, y_train, c_train, l_train, graph_train),
                (x_valid, y_valid, c_valid, l_valid, graph_valid),
                (x_test, y_test, c_test, l_test, graph_test),
                NAMES,
                SCALER,
                SCALER_y,
                le,
            ) = pickle.load(handle)
    else:
        with open(TOPORG1342_FILE, "rb") as handle:
            x, y, topo_desc, topo_class, poly_param, graph = [
                pickle.load(handle) for _ in range(6)
            ]

        # x:            graph feature
        # y:            rg2 value
        # topo_desc:    topological descriptors
        # topo_class:   topology classes
        # poly_param:   polymer generation parameters
        # graph:        networkx objects

        # preprocessing
        y = y[..., 0]

        topo_class[topo_class == "astar"] = "star"
        topo_desc = np.where(np.isnan(topo_desc), -2, topo_desc)

        le = LabelEncoder()
        topo_class = le.fit_transform(topo_class)
        NAMES = le.classes_

        # random shuffle
        x = np.random.RandomState(0).permutation(x)
        y = np.random.RandomState(0).permutation(y)
        topo_class = np.random.RandomState(0).permutation(topo_class)
        topo_desc = np.random.RandomState(0).permutation(topo_desc)
        poly_param = np.random.RandomState(0).permutation(poly_param)
        graph = np.random.RandomState(0).permutation(graph)

        # use one fold for testing
        skf = StratifiedKFold(n_splits=n_fold)
        count = -1
        for _, (train_idx, test_idx) in enumerate(skf.split(x, topo_class)):
            datasets = [x, y, topo_desc, topo_class, graph]
            train_data = [data[train_idx] for data in datasets]
            test_data = [data[test_idx] for data in datasets]

            x_train, y_train, l_train, c_train, graph_train = train_data
            x_test, y_test, l_test, c_test, graph_test = test_data

            if if_validation:
                skf2 = StratifiedKFold(n_splits=n_fold)
                train_idx2, valid_idx = next(iter(skf2.split(x_train, c_train)))
                datasets2 = [x_train, y_train, l_train, c_train, graph_train]

                x_valid, y_valid, l_valid, c_valid, graph_valid = [
                    data[valid_idx] for data in datasets2
                ]
                x_train, y_train, l_train, c_train, graph_train = [
                    data[train_idx2] for data in datasets2
                ]

            count += 1
            if count == fold:
                break

        # start adding lam
        x_train = np.repeat(x_train, repeats=n_repeat, axis=0)
        x_valid = np.repeat(x_valid, repeats=n_repeat, axis=0)
        x_test = np.repeat(x_test, repeats=n_repeat, axis=0)

        y_train = np.repeat(y_train, repeats=n_repeat)
        y_valid = np.repeat(y_valid, repeats=n_repeat)
        y_test = np.repeat(y_test, repeats=n_repeat)

        l_train = np.repeat(l_train, repeats=n_repeat, axis=0)
        l_valid = np.repeat(l_valid, repeats=n_repeat, axis=0)
        l_test = np.repeat(l_test, repeats=n_repeat, axis=0)

        c_train = np.repeat(c_train, repeats=n_repeat, axis=0)
        c_valid = np.repeat(c_valid, repeats=n_repeat, axis=0)
        c_test = np.repeat(c_test, repeats=n_repeat, axis=0)

        graph_train = np.repeat(graph_train, repeats=n_repeat)
        graph_valid = np.repeat(graph_valid, repeats=n_repeat)
        graph_test = np.repeat(graph_test, repeats=n_repeat)

        lam_train = np.random.RandomState(0).random(size=len(l_train)) * max_lambda
        lam_valid = np.random.RandomState(0).random(size=len(l_valid)) * max_lambda
        lam_test = np.random.RandomState(0).random(size=len(l_test)) * max_lambda

        nnode_train = l_train[:, 0]
        nnode_valid = l_valid[:, 0]
        nnode_test = l_test[:, 0]

        xi_train = extract_descriptors(graph_train, lam_train)
        xi_valid = extract_descriptors(graph_valid, lam_valid)
        xi_test = extract_descriptors(graph_test, lam_test)

        l_train = np.concatenate([l_train, lam_train[..., None]], axis=1)
        l_valid = np.concatenate([l_valid, lam_valid[..., None]], axis=1)
        l_test = np.concatenate([l_test, lam_test[..., None]], axis=1)

        y_train = np.concatenate(
            [
                y_train[..., None],
                nnode_train[..., None],
                xi_train[..., None],
                lam_train[..., None],
            ],
            axis=1,
        )

        y_valid = np.concatenate(
            [
                y_valid[..., None],
                nnode_valid[..., None],
                xi_valid[..., None],
                lam_valid[..., None],
            ],
            axis=1,
        )

        y_test = np.concatenate(
            [
                y_test[..., None],
                nnode_test[..., None],
                xi_test[..., None],
                lam_test[..., None],
            ],
            axis=1,
        )

        SCALER = StandardScaler()
        SCALER.fit(np.concatenate([l_train, l_valid, l_test]))
        l_train = SCALER.transform(l_train)
        l_valid = SCALER.transform(l_valid)
        l_test = SCALER.transform(l_test)

        SCALER_y = MinMaxScaler(feature_range=(-1, 1))
        SCALER_y.fit(np.concatenate([y_train, y_valid, y_test]))
        y_train = SCALER_y.transform(y_train)
        y_valid = SCALER_y.transform(y_valid)
        y_test = SCALER_y.transform(y_test)

        # random shuffle
        x_train = np.random.RandomState(0).permutation(x_train)
        x_valid = np.random.RandomState(0).permutation(x_valid)

        y_train = np.random.RandomState(0).permutation(y_train)
        y_valid = np.random.RandomState(0).permutation(y_valid)

        l_train = np.random.RandomState(0).permutation(l_train)
        l_valid = np.random.RandomState(0).permutation(l_valid)

        c_train = np.random.RandomState(0).permutation(c_train)
        c_valid = np.random.RandomState(0).permutation(c_valid)

        graph_train = np.random.RandomState(0).permutation(graph_train)
        graph_valid = np.random.RandomState(0).permutation(graph_valid)
        # end adding lam

        with open(data_dir, "wb") as handle:
            pickle.dump(
                (
                    (x_train, y_train, c_train, l_train, graph_train),
                    (x_valid, y_valid, c_valid, l_valid, graph_valid),
                    (x_test, y_test, c_test, l_test, graph_test),
                    NAMES,
                    SCALER,
                    SCALER_y,
                    le,
                ),
                handle,
            )

    if if_validation:
        if verbose:
            print(f"Train: {len(x_train)} Valid: {len(x_valid)} Test: {len(x_test)}")
        return (
            (x_train, y_train, c_train, l_train, graph_train),
            (x_valid, y_valid, c_valid, l_valid, graph_valid),
            (x_test, y_test, c_test, l_test, graph_test),
            NAMES,
            SCALER,
            SCALER_y,
            le,
        )

    else:
        if verbose:
            print(f"Train: {len(x_train)} Test: {len(x_test)}")
        return (
            (x_train, y_train, c_train, l_train, graph_train),
            (x_test, y_test, c_test, l_test, graph_test),
            NAMES,
            SCALER,
            SCALER_y,
            le,
        )
