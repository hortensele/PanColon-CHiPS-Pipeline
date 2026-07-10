#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --job-name=ssl_remove
#SBATCH --output=ssl_remove_%A_%a.out
#SBATCH --error=ssl_remove_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --mem=1GB

module load condaenvs/gpu/pathgan_SSL37

python3 remove_indexes_h5.py --h5_file /gpfs/data/tsirigoslab/home/leh06/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/nyuimmuno_5x/h224_w224_n3_zdim128/hdf5_nyuimmuno_5x_he_train.h5 --pickle_file /gpfs/data/tsirigoslab/home/leh06/Histomorphological-Phenotype-Learning/utilities/files/indexes_to_remove/v07_10panCancer_5x/nyuimmuno_5x.pkl