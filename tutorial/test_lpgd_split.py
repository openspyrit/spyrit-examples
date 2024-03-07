import datetime
import subprocess
import os

# Parameters
# (the first three paramaters allow to generalize train_gen_meas.py 
# to common measurement types)
meas = 'hadam-split'    # measurement type
noise = 'poisson' # noise type
prep = 'split-poisson'    # preprocessing type
#
N0 = 10.0        # ph/pixel max: number of counts
img_size = 64   # image size
M =  img_size**2 // 4  # Num measurements = subsampled by factor 4
subs = 'rect' # subsampling types: 'var': high variance, 'rect': low frequency
data_root = '../../data/'  # data root path (where the data is downloaded)
data = 'stl10' #'stl10', 'imagenet' 
#stat_root = './stat/'  # stat root path (where the cov is stored, required for split meas)
stat_root = '../../stat/ILSVRC2012_v10102019/'
#data_root = '../../data/ILSVRC2012_v10102019/'
#
arch = 'lpgd' # Network architecture:  'upgd', 'lpgd
unfold_iter = 2 # Number of UPGD iterations
denoi = 'unet' # Denoiser architecture
num_epochs = 30
batch_size = 128
# Path to previous trained model
checkpoint_model = ""
#checkpoint_model = './model/lpgd_unet_stl10_N0_10_m_hadam-split_N_64_M_1024_epo_1_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07.pth' 
checkpoint_interval = 10     # Interval to save the model
# Learneable stepsize
unfold_step_grad = False
#
# Tensorboard logs path
name_run = f"{data}_splitmeas_{subs}_M{M}_N{int(N0)}_{img_size}x{img_size}_{arch}_{denoi}_iter{unfold_iter}"
if checkpoint_model != '':
    name_run += '_cont'
mode_tb = True
if (mode_tb is True):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    tb_path = f'../../runs/runs_{name_run}/{now}'
    print(f"Tensorboard logdir {tb_path}")
else:
    tb_path = ""    
tb_prof = False # False

print(f"Current path: {os.getcwd()}")

#os.chdir('spyrit-examples/tutorial')

# Run train.py
#!python3 train_gen_meas.py --meas 'hadam-pos' --noise 'no-noise' --prep 'dir-poisson'--N0 1 --M 1024 --data_root './data/' --data 'stl10' --stat_root '' --tb_path 'runs/hadam-pos_no-noise_1' --arch 'pinv-net' --denoi 'cnn' --num_epochs 30
#subprocess.run(['python3', 'train_gen_meas.py', '--tb_path', tb_path])
subprocess.run(['python3', 'train_gen_meas.py', '--meas', meas, '--noise', noise, '--prep', prep,
                '--data_root', data_root, '--data', data, '--stat_root', stat_root,
                '--N0', str(N0), '--M', str(M), '--subs', subs, '--img_size', str(img_size),
                '--arch', arch, '--denoi', denoi, '--num_epochs', str(num_epochs),
                '--batch_size', str(batch_size),
                '--unfold_iter', str(unfold_iter),
                '--unfold_step_grad', str(unfold_step_grad),
                '--tb_path', tb_path,
                '--checkpoint_interval', str(checkpoint_interval), 
                '--checkpoint_model', checkpoint_model])

# Get name of the current python file
filename = os.path.basename(__file__)

# Copy current python file to the log directory tb_path
subprocess.run(['cp', filename, tb_path])