
# Data-Driven and Physics-Guided Design of Viscosity-Modifying Polymers

<br />
<img src="./fig/toc.png" />
<br />



## Install Instructions

The `setup.py` file contains all the necessary packages to reproduce the results. 

```console
$ git clone https://github.com/webbtheosim/cg-topo-solv
$ cd cg-topo-solv
$ conda create --name py310hoomd511 python=3.10.16 -y
$ conda activate py310hoomd511
$ # or source activate py310hoomd511
$ pip install -r requirements.txt
$ pip install -e .
```

### Simulation Installation

If you want to install the simulation packages and rerun the MPCD simulation with the reverse-perturbation method, you first need HOOMD-blue 5.1.1 and AZPlugins 1.1.0. You can install them by following each project’s build guide。

1. Follow HOOMD-blue’s build instructions for version 5.1.1:  
   https://hoomd-blue.readthedocs.io/en/v5.1.1/building.html

2. Follow AZPlugins’ build instructions for version 1.1.0:  
   https://azplugins.readthedocs.io/en/v1.1.0/index.html


## File Structure

`cg_topo_solv/`  
Core source code:

- `analysis/`  
  Analysis tools for viscosity, Carreau–Yasuda fitting, and polymer conformations.

- `bo/`  
  Bayesian optimization with target-aware expected improvement and active learning.

- `ml/`  
  Variational autoencoder for learning a chemically and topologically meaningful latent space of polymers.

- `plot/`  
  Plotting utilities.

`notebook/`  
Jupyter notebooks reproducing manuscript figures.

`submit/`  
Slurm job submission scripts for HPC clusters.

## Download Data and Results

Please download [data]

```console
$ conda activate py310hoomd511
$ pip install gdown
$ cd cg-topo-solv
$ gdown "1EBHWSdO5ODlUYefj_e6yARHJZAfj3OHy"
$ tar -xzvf mpcd_result.tar
$ gdown "1hX3Gce7Bb_121Go26B7mLM9c5OIBNIVZ"
$ tar -xzvf data_ml.tar
$ gdown "1w_hgf42OPuyaPA1D9h6XnyS_1URt6bFf"
$ tar -xzvf mpcd_ml_weight.tar
$ gdown "19_13RAAzu5-ZOKBqRk6wkBGF9rgBYuR4"
$ tar -xzvf mpcd_ml_analysis.tar
$ gdown "1HQWmemKO-TvSehEHVD43E1Li16yBP-SB"
$ tar -xzvf topo_data.tar

`mpcd_result/`  
Contains all results necessary to reproduce the plots, including:
- Viscosity simulation outputs
- Latent representations of all proposed polymer structures

`data_ml/`  
Contains the `topo-solv-6k` dataset — a topologically and chemically diverse polymer set used for training the VAE and extracting latent representations.

`topo_data/`  
Original dataset from which `topo-solv-6k` was derived. Useful for users interested in applying different data augmentation or filtering strategies.

`mpcd_ml_analysis/`  
Contains latent representations sampled from:
- The initial seed dataset
- Space-filling algorithms used during exploration

`mpcd_ml_weight/`  
Contains 500 model weight files from hyperparameter tuning.  
The optimal model selected from this set is used for polymer generation.
