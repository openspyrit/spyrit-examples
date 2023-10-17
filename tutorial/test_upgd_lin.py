import datetime
import subprocess
import os

# Parameters
# (the first three paramaters allow to generalize train_gen_meas.py 
# to common measurement types)
meas = 'hadam-pos'    # measurement type
noise = 'poisson' # noise type
prep = 'dir-poisson'    # preprocessing type
#
N0 = 100.0        # ph/pixel max: number of counts
img_size = 64   # image size
M =  img_size**2 // 4  # Num measurements = subsampled by factor 4
subs = 'rect' # subsampling types: 'var': high variance, 'rect': low frequency
data_root = './data/'  # data root path (where the data is downloaded)
data = 'stl10'
#
arch = 'upgd' # Network architecture
upgd_iter = 1 # Number of UPGD iterations
denoi = 'cnn' # Denoiser architecture
num_epochs = 5
#checkpoint_model = './model' # Path to previous trained model
checkpoint_interval = 1     # Interval to save the model
#
# Tensorboard logs path
name_run = "stdl10_hadpos_upgd_1ter_lapos_tr_after2epoch"
mode_tb = True 
if (mode_tb is True):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    tb_path = f'runs/runs_{name_run}_n{int(N0)}_m{M}/{now}'
else:
    tb_path = ""    
tb_prof = False # False

print(f"Current path: {os.getcwd()}")

#os.chdir('spyrit-examples/tutorial')

# Run train.py
#!python3 train_gen_meas.py --meas 'hadam-pos' --noise 'no-noise' --prep 'dir-poisson'--N0 1 --M 1024 --data_root './data/' --data 'stl10' --stat_root '' --tb_path 'runs/hadam-pos_no-noise_1' --arch 'pinv-net' --denoi 'cnn' --num_epochs 30
#subprocess.run(['python3', 'train_gen_meas.py', '--tb_path', tb_path])
subprocess.run(['python3', '~/Projects/openspyrit/spyrit-examples/tutorial/train_gen_meas.py', '--meas', meas, '--noise', noise, '--prep', prep,
                '--data_root', data_root, '--data', data, 
                '--N0', str(N0), '--M', str(M), '--subs', subs, '--img_size', str(img_size),
                '--arch', arch, '--denoi', denoi, '--num_epochs', str(num_epochs),
                '--upgd_iter', str(upgd_iter),
                '--tb_path', tb_path,
                '--checkpoint_interval', str(checkpoint_interval)])