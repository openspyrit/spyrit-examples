#!/bin/bash
DAT_PATH='../../datasets/ILSVRC2012_v10102019/'

# for expe patterns
PAT_PATH='./data/2023_03_13_2023_03_14_eGFP_DsRed_3D/Reconstruction/Mat_rc/motifs_Hadamard_128_512.npy'

# for Hadam1Split patterns
M=128
N=512
A=250 # number of photons

E=20    # number of epochs
B=20    # batch size

# tikho-net
#python train_net.py --arch tikho-net --M $M --img_size $N --data_root $DAT_PATH --batch_size $B --num_epochs $E --alpha $A
python train_net.py --pattern_root $PAT_PATH --arch tikho-net --data_root $DAT_PATH --batch_size $B --num_epochs $E --alpha $A

# pinv-net
#python train_net.py --M $M --img_size $N --data_root $DAT_PATH --batch_size $B --num_epochs $E --alpha $A
python train_net.py --pattern_root $PAT_PATH --data_root $DAT_PATH --batch_size $B --num_epochs $E --alpha $A