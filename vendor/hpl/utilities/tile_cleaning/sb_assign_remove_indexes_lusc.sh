#!/bin/bash
#SBATCH --partition=cpu_medium
#SBATCH --job-name=ssl_remove
#SBATCH --output=ssl_remove_%A_%a.out
#SBATCH --error=ssl_remove_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=1GB

module load condaenvs/gpu/pathgan_SSL37

python3 remove_indexes_h5.py --h5_file /gpfs/data/tsirigoslab/home/leh06/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/tcga_lung_20x/h224_w224_n3_zdim128/hdf5_tcga_lung_20x_he_train.h5 --pickle_file /gpfs/data/tsirigoslab/home/leh06/Histomorphological-Phenotype-Learning/utilities/files/indexes_to_remove/tcga_lung_20x/tcga_lung_20x_he_train_luad.pkl
