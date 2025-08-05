import numpy as np
import tensorflow as tf
import networkx as nx

from tqdm import tqdm
from scipy.optimize import minimize
from scipy.stats import ncx2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K

from cg_topo_solv.analysis.graph import get_desc
from cg_topo_solv.ml.clean import graph_anneal_break_largest_circle


def target_value(params_fit, params_target, eta_fit, eta_target, mode="param"):
    """Return training and target values for parameter or viscosity-based optimization."""

    if mode == "param":
        y_train = np.array(params_fit)
        y_target = np.array(params_target)
        y_train[:, 2] = np.log10(y_train[:, 2])
        y_target[:, 2] = np.log10(y_target[:, 2])

    elif mode == "eta":
        y_train = np.array(eta_fit)
        y_target = np.array(eta_target)

    return y_train, y_target



def process_batch(model, x_batch, SCALER_y, SCALER, batch_size=256):
    """Process a batch of latent vectors through decoder, regressor, and encoder pipeline."""
    
    K.clear_session()

    decoder = model.get_layer("decoder")
    regressor = model.get_layer("regressor")

    d_in = tf.keras.Input(shape=(8,), name="gen_d_in")

    x = d_in
    for layer in decoder.layers:
        x = layer(x)
    gen_model = tf.keras.Model(inputs=d_in, outputs=x)

    x = d_in
    for layer in regressor.layers:
        x = layer(x)
    regressor_model = tf.keras.Model(inputs=d_in, outputs=x)

    sampled_data = gen_model.predict(x_batch, verbose=0)
    sampled_labels = SCALER_y.inverse_transform(
        regressor_model.predict(x_batch, verbose=0)
    )

    Gs = []
    topo_descs = []
    adjs = []

    for i in range(len(x_batch)):
        G = graph_anneal_break_largest_circle(sampled_data[i])
        topo_desc = get_desc(G)
        topo_desc = np.concatenate((topo_desc, sampled_labels[i][-1][None, ...]))[None, ...]
        topo_desc = SCALER.transform(topo_desc)

        adj = nx.to_numpy_array(G)
        adj = np.pad(adj, ((0, 100 - adj.shape[0]), (0, 100 - adj.shape[0])), "constant")[None, ...]

        Gs.append(G)
        topo_descs.append(topo_desc)
        adjs.append(adj)

    x_in = model.get_layer("x_in").input
    a_1 = model.get_layer("a_1").input
    f_1 = model.get_layer("f_1").input
    z1 = model.get_layer("z1").output
    z2 = model.get_layer("z2").output

    encoder_model = tf.keras.Model(inputs=[f_1, a_1, x_in], outputs=[z1, z2])

    adjs = np.vstack(adjs)
    topo_descs = np.vstack(topo_descs)
    z1_vals, z2_vals = encoder_model.predict([adjs, adjs, topo_descs], verbose=0)

    return Gs, sampled_labels, topo_descs, z1_vals, z2_vals


def distance(y_pred, y_target):
    """Compute mean squared error between predicted and target values."""
    return mean_squared_error(y_pred.flatten(), y_target.flatten())


def loss_function(x, gp, y_target):
    """Compute loss as mean squared error between GP prediction and target."""
    y_pred, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    y_pred = y_pred.flatten()
    y_target = y_target.flatten()

    integral_diff = distance(y_pred, y_target)

    return integral_diff


def target_driven_ei(x, gp, y_target, best_delta, xi=0.0):
    """Compute target-driven expected improvement using noncentral chi-squared statistics."""
    mu, std = gp.predict(x.reshape(1, -1), return_std=True)
    mu = mu.flatten()
    sigma = std.flatten()
    gamma2 = np.mean(sigma**2)

    lambda_par = np.sum((mu - y_target)**2) / gamma2
    target = best_delta - xi

    if target <= 0:
        return 0.0

    u = target / gamma2
    d = len(mu)

    Fd = ncx2.cdf(u, d, lambda_par)
    Fd2 = ncx2.cdf(u, d + 2, lambda_par)
    Fd4 = ncx2.cdf(u, d + 4, lambda_par)

    return float(target * Fd - gamma2 * (d * Fd2 + lambda_par * Fd4))


def propose_next_point(gp, y_target, best_delta, bounds, x0, xi=0.0):
    """Propose next point by maximizing target-driven expected improvement."""
    def acq(x):
        return -target_driven_ei(x, gp, y_target, best_delta, xi)
    
    return minimize(acq, x0=x0, bounds=bounds, method='L-BFGS-B')


def whats_next(
    model,
    x_train,
    y_train,
    y_target,
    bounds,
    SCALER_y,
    SCALER,
    gp=None,
    max_attempts=10000,
    distance_threshold=0.01,
    batch_size=100,
):
    """Suggest the next sample point using a GP model and target-driven EI acquisition."""
    
    if gp is None:
        nu_values = np.round(np.arange(1.0, 2.6, 0.1), 1)
        best_nu = None
        best_score = np.inf

        for nu in nu_values:
            kernel = Matern(nu=nu)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []

            for train_index, test_index in kf.split(x_train):
                x_fold_train, x_fold_test = x_train[train_index], x_train[test_index]
                y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
                gp.fit(x_fold_train, y_fold_train)
                fold_loss = sum(loss_function(x, gp, y_target) for x in x_fold_test)
                scores.append(fold_loss / len(x_fold_test))

            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_nu = nu

        print(f"Best nu value: {best_nu}, with cross-validated loss: {best_score:0.3f}")

        gp = GaussianProcessRegressor(kernel=Matern(nu=best_nu), n_restarts_optimizer=20, normalize_y=True)
        gp.fit(x_train, y_train)

    losses = [loss_function(x, gp, y_target) for x in x_train]
    best_loss = np.min(losses)
    print("best loss", np.round(best_loss, 3), np.argmin(losses))

    best_distance = float("inf")
    best_result = None

    for i in range(0, max_attempts, batch_size):
        current_batch_size = min(batch_size, max_attempts - i)
        x_batch = []

        top_indices = np.argsort(losses)[:5]
        for _ in range(current_batch_size):
            idx = np.random.choice(top_indices)
            result = propose_next_point(
                gp,
                y_target,
                best_loss,
                bounds,
                x0=x_train[idx] + np.random.normal(0, 2.0, size=(8,))
            )
            x_batch.append(result.x)

        x_batch = np.array(x_batch)

        try:
            Gs, sampled_labels, topo_descs, z1_vals, z2_vals = process_batch(
                model, x_batch, SCALER_y, SCALER, batch_size
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

        for j in tqdm(range(current_batch_size)):
            distance = np.abs(z1_vals[j] - x_batch[j]).mean()

            if distance < best_distance:
                best_distance = distance
                gp_pred = gp.predict(x_batch[j].reshape(1, -1))
                best_result = (
                    Gs[j],
                    sampled_labels[j],
                    gp_pred,
                    topo_descs[j],
                    z1_vals[j],
                    z2_vals[j],
                    x_batch[j],
                )

                print(f"Attempt {i + j + 1}, distance: {distance:.3f}")

            if distance < distance_threshold:
                print(f"Found good match on attempt {i + j + 1}")
                G, sampled_label, gp_pred, topo_desc, z1_val, z2_val, x_next = best_result
                print("\nFinal results:")
                print("Next point to evaluate:", np.round(x_next, 2))
                print("After graph cleaning:", np.round(z1_val.squeeze(), 2))
                print("Final distance:", best_distance)
                return (
                    G,
                    sampled_label,
                    topo_desc,
                    z1_val,
                    x_train[np.argmin(losses)][None, ...],
                    gp,
                )

    if best_result is None:
        raise ValueError("Could not find a good match after maximum attempts")

    G, sampled_label, gp_pred, topo_desc, z1_val, z2_val, x_next = best_result
    print("\nFinal results:")
    print("Next point to evaluate:", np.round(x_next, 2))
    print("After graph cleaning:", np.round(z1_val.squeeze(), 2))
    print("Final distance:", best_distance)

    return (
        G,
        sampled_label,
        topo_desc,
        z1_val,
        x_train[np.argmin(losses)][None, ...],
        gp,
    )
    
    
def calculate_distance(y_pred, y_target, mode="mae"):
    if mode == "mae":
        return mean_absolute_error(y_target.flatten(), y_pred.flatten())
    elif mode == "mse":
        return mean_squared_error(y_target.flatten(), y_pred.flatten())


def check_bayesian_performance(y_trains, y_targets, mode="mae", k=5):
    top_k_distances = []
    for y_target in y_targets:
        distances = [
            calculate_distance(y_train, y_target, mode) for y_train in y_trains
        ]
        # Sort the distances and select the top k smallest distances.
        sorted_distances = sorted(distances)
        top_k = sorted_distances[:k]
        top_k_distances.append(top_k)
    return np.array(top_k_distances)
