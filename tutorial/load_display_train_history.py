import os
import pickle
import matplotlib.pyplot as plt

# Specify the .pkl file path
file_path = '../../model/TRAIN_lpgd_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_15_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07_uit_3.pkl'  # Replace with your .pkl file path
#file_path = "../../model/TRAIN_pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_15_lr_0.001_sss_10_sdr_0.5_bs_128_reg_1e-07.pkl"
#file_path = "../../model/TRAIN_lpgd_cnn_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_1_lr_0.001_sss_10_sdr_0.5_bs_256_reg_1e-07_uit_3.pkl"

# Save path
save_path = '../../results/'
name_save = os.path.basename(file_path).replace('.pkl', '').replace('.', '-')

# Open the file in read-binary mode and load the data
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    train_loss = data.train_loss
    val_loss = data.val_loss

# Print the loaded data
fig = plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.legend()
fig.savefig(os.path.join(save_path, name_save + '.png'),  bbox_inches='tight', dpi=120)

