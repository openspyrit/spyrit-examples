#!/bin/bash
DAT_PATH='../../datasets/ILSVRC2012_v10102019/'
# for expe patterns
PAT_PATH='/d/save_phd/PhD/python/PTND/Data/eGFP_DsRed_3D/Reconstruction/Mat_rc/motifs_Hadamard_128_512.npy'
# for Hadam1Split patterns
M=128
N=512
E=30    # number of epochs
B=20    # batch size

python train_net.py --M $M --img_size $N --data_root $DAT_PATH --batch_size $B --num_epochs $E 
python train_net.py --pattern_root $PAT_PATH --data_root $DAT_PATH --batch_size $B --num_epochs $E