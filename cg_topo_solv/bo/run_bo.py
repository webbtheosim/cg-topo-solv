import os
import re
import time
import glob
import pickle
import argparse

import numpy as np
import networkx as nx
from matplotlib import cm, colors as mcolors
import proplot as pplt
from tensorflow.keras import backend as K

from cg_topo_solv.ml.data import load_data
from cg_topo_solv.ml.latent_sampling import latent_sampling
from cg_topo_solv.ml.train import train_vae
from cg_topo_solv.ml.vae import get_file_names
from cg_topo_solv.bo.propose import target_value, whats_next


################ USER CAN ADJUST DIR ################
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHT_DIR = BASE_DIR / "mpcd_ml_weight"
ANALYSIS_DIR = BASE_DIR / "mpcd_ml_analysis"
DATA_FILE = BASE_DIR / "data_ml" / "data_aug20.pickle"
VISCO_DATA_DIR = BASE_DIR / "mpcd" / "result_1106"
################ USER CAN ADJUST DIR ################

def parse_args():
    parser = argparse.ArgumentParser(description="Run VAE training and BO sampling for copolymer design.")
    parser.add_argument("--job_iter", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--mode", type=str, default="al")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--scratch_dir", type=str, default="/scratch/")
    parser.add_argument("--home_dir", type=str, default="/home/")
    parser.add_argument("--weight_dir", type=str, default=str(WEIGHT_DIR))
    parser.add_argument("--figure_dir", type=str, default=str(BASE_DIR / "mpcd_figure"))
    parser.add_argument("--ml_data", type=str, default=str(DATA_FILE))
    return parser.parse_args()


def main(job_index, mode="al", iteration=1, rerun=False,
         SCRATCH_DIR="/scratch/",
         HOME_DIR="/home/",
         WEIGHT_DIR=str(WEIGHT_DIR),
         FIGURE_DIR=str(BASE_DIR / "mpcd_figure"),
         ML_DATA=str(DATA_FILE)):
    """Main driver for AL iteration, training, sampling, and batch result aggregation."""

    RESULT_DIR = os.path.join(HOME_DIR, "mpcd_result")
    SCRATCH_RESULT_DIR = os.path.join(SCRATCH_DIR, "mpcd", "result_1106")

    FILE_VISC_SEED = os.path.join(RESULT_DIR, "fit_visc_seed.pickle")
    FILES_al = [FILE_VISC_SEED] + [
        os.path.join(RESULT_DIR, f"fit_visc_al_{i}.pickle") for i in range(1, 5)
    ]

    FILES_SAMPLE = [
        os.path.join(SCRATCH_RESULT_DIR, f"batch{i}al", f"batch{i}_sample_al.pickle")
        for i in range(1, 5)
    ]

    FILE_TARGET = os.path.join(RESULT_DIR, "target_curves.pickle")

    FILE_OUTPUT = os.path.join(
        SCRATCH_RESULT_DIR,
        f"batch_{iteration}al",
        f"batch{iteration}_sample_al_{job_index}.pickle",
    )

    FILE_OUTPUT_0 = os.path.join(
        SCRATCH_RESULT_DIR,
        f"batch_{iteration}al",
        f"batch_{iteration}_sample_al_0.pickle",
    )

    if not os.path.exists(FILE_OUTPUT) or rerun:

        sample, z_mean, y_all, c_all, graph_all = latent_sampling(
            seed=42, mode="maximin_sampling", rerun=False, size=30
        )

        if iteration > 1:
            zs = []
            for file_sample in FILES_SAMPLE[:iteration - 1]:
                with open(file_sample, "rb") as handle:
                    _ = pickle.load(handle)
                    _ = pickle.load(handle)
                    _ = pickle.load(handle)
                    z1_list = pickle.load(handle)
                zs.append(z1_list)
            zs = np.concatenate(zs, axis=0)

        if iteration == 1:
            x_train = np.copy(np.array(z_mean[sample])).squeeze()
        else:
            x_train = np.copy(np.array(z_mean[sample])).squeeze()
            x_train = np.concatenate([x_train, zs], axis=0)

        x_train_new = x_train
        bounds = np.array([[-4] * 8, [4] * 8]).T

        with open(FILE_TARGET, "rb") as f:
            loaded_data = pickle.load(f)
        params = loaded_data["params"]
        eta_targets = loaded_data["curves"]

        yy = []
        ps = []
        for file_visc in FILES_al[:iteration]:
            with open(file_visc, "rb") as handle:
                _, yy_, _, ps_ = (pickle.load(handle) for _ in range(4))
            yy.append(yy_)
            ps.append(ps_)
        yy = np.concatenate(yy, axis=0)
        ps = np.concatenate(ps, axis=0)

        y_train_new, y_target_new = target_value(
            params_fit=ps,
            params_target=params,
            eta_fit=yy,
            eta_target=eta_targets,
            mode="eta",
        )

        print(y_train_new.shape, y_target_new.shape)

        results = {
            "Gs": [],
            "y_preds": [],
            "gp_preds": [],
            "z1_list": [],
            "z_data_list": [],
            "desc_list": [],
        }

        (
            (x_train, y_train, c_train, l_train, graph_train),
            (x_valid, y_valid, c_valid, l_valid, graph_valid),
            (x_test, y_test, c_test, l_test, graph_test),
            NAMES,
            SCALER,
            SCALER_y,
            LE,
        ) = load_data(
            data_dir=ML_DATA,
            fold=0,
            if_validation=True,
            rerun=False,
        )

        has_nan = np.isnan(l_train).any()
        has_inf = np.isinf(l_train).any()

        print("l_train contains NaN values:", has_nan)
        print("l_train contains inf values:", has_inf)
        print(f"l_train shape: {l_train.shape}")

        K.clear_session()

        LATENT_DIM = 8
        DECODER = "cnn"
        ENCODER = "desc_gnn"
        MONITOR = "val_decoder_acc"
        IF_REG = True
        IF_CLS = True
        LR = 0.005
        BS = 256
        rw = 100.0
        cw = 100.0

        weight_name, hist_name = get_file_names(
            ENCODER, DECODER, "20240816", LATENT_DIM, MONITOR,
            IF_REG, IF_CLS, 1.0, [1.0, rw, cw], LR, BS
        )

        print(weight_name)

        if_train = not os.path.exists(os.path.join(WEIGHT_DIR, hist_name))
        print(if_train)

        model, pickle_file = train_vae(
            ENCODER, DECODER, MONITOR, IF_REG, IF_CLS,
            x_train, x_valid, y_train, y_valid,
            c_train, c_valid, l_train, l_valid,
            1.0, [1.0, rw, cw],
            LR, BS, if_train, verbose=2, n_epoch=1000
        )

        if job_index != 0:
            if os.path.exists(FILE_OUTPUT_0):
                with open(FILE_OUTPUT_0, "rb") as handle:
                    _ = pickle.load(handle)
                    _ = pickle.load(handle)
                    gp_model = pickle.load(handle)
            else:
                while not os.path.exists(FILE_OUTPUT_0):
                    print("Waiting for the first job to finish...")
                    time.sleep(30)
                with open(FILE_OUTPUT_0, "rb") as handle:
                    _ = pickle.load(handle)
                    _ = pickle.load(handle)
                    gp_model = pickle.load(handle)

        current_gp = None if job_index == 0 else gp_model[-1]

        G, y_pred, desc, z1, z_data, gp = whats_next(
            model, x_train_new, y_train_new, y_target_new[job_index],
            bounds, SCALER_y, SCALER, gp=current_gp, max_attempts=10000
        )

        results["Gs"].append(G)
        results["y_preds"].append(y_pred)
        results["gp_preds"].append(gp)
        results["z1_list"].append(z1)
        results["z_data_list"].append(z_data)
        results["desc_list"].append(desc)

        print(f"Iteration {job_index + 1}/{len(y_target_new)} completed")

        with open(FILE_OUTPUT, "wb") as handle:
            for data in results.values():
                pickle.dump(data, handle)

        if job_index == len(y_target_new) - 1:
            FILE_OUTPUT_PATTERN = os.path.join(
                SCRATCH_RESULT_DIR,
                f"batch_{iteration}al",
                f"batch_{iteration}_sample_al_*.pickle"
            )

            while True:
                files = glob.glob(FILE_OUTPUT_PATTERN)
                files_sorted = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.pickle', x).group(1)))
                if len(files_sorted) == len(y_target_new):
                    break
                print("Waiting for all files to be ready...")
                time.sleep(30)

            Gs_all, y_preds_all, gp_preds_all, z1_all, z_data_all, desc_all = [], [], [], [], [], []

            for file in files_sorted:
                with open(file, "rb") as handle:
                    Gs, y_preds, gp_preds, z1_list, z_data_list, desc_list = (
                        pickle.load(handle) for _ in range(6)
                    )
                Gs_all.extend(Gs)
                y_preds_all.extend(y_preds)
                gp_preds_all.extend(gp_preds)
                z1_all.extend(z1_list)
                z_data_all.extend(z_data_list)
                desc_all.extend(desc_list)

            FILE_OUTPUT_GLOBAL = os.path.join(
                SCRATCH_RESULT_DIR,
                f"batch_{iteration}al",
                f"batch{iteration}_sample_al.pickle"
            )

            with open(FILE_OUTPUT_GLOBAL, "wb") as f:
                pickle.dump(Gs_all, f)
                pickle.dump(y_preds_all, f)
                pickle.dump(gp_preds_all, f)
                pickle.dump(z1_all, f)
                pickle.dump(z_data_all, f)
                pickle.dump(desc_all, f)
                pickle.dump(files_sorted, f)

            cmap = cm.coolwarm
            norm = mcolors.Normalize(vmin=0.0, vmax=0.5)

            fig, ax = pplt.subplots(nrows=6, ncols=5, refwidth=1, refheight=1, wspace=0, hspace=1.2)

            for i in range(30):
                np.random.seed(42)
                pos = nx.kamada_kawai_layout(Gs_all[i])
                pos = nx.spring_layout(Gs_all[i], pos=pos, iterations=200)

                lambda_value = y_preds_all[i][-1]
                node_color = cmap(norm(lambda_value))

                nx.draw_networkx_nodes(
                    Gs_all[i], pos,
                    node_color=[node_color] * len(Gs_all[i].nodes),
                    node_size=10,
                    edgecolors="black",
                    linewidths=0.2,
                    ax=ax[i]
                )
                nx.draw_networkx_edges(Gs_all[i], pos, edge_color="k", ax=ax[i])
                ax[i].axis("off")
                ax[i].set_title(r"$\mathit{\lambda}=$" + f"{lambda_value:0.3f}", size=12)

            fig.save(
                os.path.join(FIGURE_DIR, f"batch_{iteration}_{mode}_graph.png"),
                dpi=300,
                background="white"
            )

        return results

    else:
        print("File already exists, skipping computation.")
        print(FILE_OUTPUT)


if __name__ == "__main__":
    args = parse_args()

    main(
        job_index=args.job_index,
        iteration=args.iteration,
        mode=args.mode,
        rerun=args.rerun,
        SCRATCH_DIR=args.scratch_dir,
        HOME_DIR=args.home_dir,
        WEIGHT_DIR=args.weight_dir,
        FIGURE_DIR=args.figure_dir,
        ML_DATA=args.ml_data
    )
