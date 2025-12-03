# GoFlow
This repository contains a minimal version of GoFlow which makes it particularly easy to get up and running. GoFlow is an open-source model for predicting transition state geometries of single-step organic reactions.

# Installation

Create environment, activate, and install GoFlow in editable mode
```bash
conda env create -f environment.yml
conda activate goflow
pip install -e .
```

Install PyTorch & related
```bash
# Linux
pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' 'torchvision==0.21.0'
pip install -f https://data.pyg.org/whl/torch-2.6.0+cu124.html pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# MacOS
pip install torch==2.6.0 torchvision==0.21.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

## Configuration
We use [Hydra](https://hydra.cc/) for managing model configurations and experiments.

All hyper-parameters are found in the configs directory and its subdirectories (`./configs`).

GoFlow is trained and evaluated on the open-source [RDB7 database](https://zenodo.org/records/13328872) by [Spiekermann et al.](https://www.nature.com/articles/s41597-022-01529-6). The raw `.csv` and `.xyz` files are located in the `data/RDB7/raw_data` directory.

Preprocess the dataset by running the `preprocess_rdb7.sh` script, which produces `.pkl` files containing the split indices and the `data.pkl` file. Make sure to adjust the paths to the `.csv` and `.xyz` files inside the script as needed.

The processed data, i.e., each reaction, is stored as a [PyG](https://pytorch-geometric.readthedocs.io/) object in a Python list and is located in the `data/RDB7/processed_data` directory as `data.pkl`.

## Usage
Each experiment has a separate shell script (.sh files) in the `scripts` folder.

- To train the model, run the `train_rdb7.sh` script.

- To test the model, first run the `test_save_all_samples_rdb7.sh` script, which performs inference on the test set.

- To compute evaluation metrics, run the `test_samples_analysis_rdb7.sh` script with the required input and output file arguments.

Modify the shell scripts as required to set custom paths for your input and output directories. Also, edit the configuration files as needed.

## Acknowledgement

GoFlow is built upon open-source code provided by [TsDiff](https://github.com/seonghann/tsdiff) and [GotenNet](https://github.com/sarpaykent/GotenNet).

## License
Our model and code are released under MIT License.

## Cite

If you use this code in your research, please cite the following paper:

```bibtex
@Article{galustian2025goflow,
author="Galustian, Leonard and Mark, Konstantin and Karwounopoulos, Johannes and Kovar, Maximilian P.-P. and Heid, Esther",
title="GoFlow: efficient transition state geometry prediction with flow matching and E(3)-equivariant neural networks",
journal="Digital Discovery",
year="2025",
pages="-",
publisher="RSC",
doi="10.1039/D5DD00283D",
url="http://dx.doi.org/10.1039/D5DD00283D",
}
```
