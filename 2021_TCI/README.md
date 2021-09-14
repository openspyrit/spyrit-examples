#Train
##STL-10

```
python scripts/tci/train_nSDCAN_Conv.py --CR 512 --N0 10 --sig 0 --n_iter 0 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --data_val_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --net_arch 0 --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI/Unet/" --expe_root "data/expe/" --num_epochs 100 --batch_size 64 --step_size 10
```

```
python scripts/tci/train_noisy_SDCAN.py --CR 512 --N0 10 --sig 0 --model_recon 'U_net' --n_iter 0 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --data_val_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 0 --model_root "models/TCI/Unet/" --expe_root "data/expe/" --num_epochs 100 --batch_size 64 --step_size 10
```

```
python scripts/tci/train_nSDCAN_Conv.py --CR 512 --N0 10 --sig 0 --n_iter 0 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --data_val_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI/Unet/" --expe_root "data/expe/" --num_epochs 100 --batch_size 64 --step_size 10
```

```
python scripts/tci/train_neumann_bis.py --CR 512 --N0 10 --sig 0 --n_iter 4 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --data_val_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 0 --model_root "models/TCI/neumann/" --expe_root "data/expe/" --num_epochs 60 --batch_size 64 --reg 0.00001 --step_size 10 --checkpoint_model "models/TCI/Unet/U_net_pinv_N0_10_sig_0_N_64_M_512_epo_100_lr_0.001_sss_10_sdr_0.5_bs_64_reg_1e-07.pth"
```

```
python scripts/tci/train_modl_2.py --CR 512 --N0 10 --sig 0 --n_iter 4 --poisson_dc 1 --data_train_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --data_val_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI/modl/" --expe_root "data/expe/" --num_epochs 40 --batch_size 64 --step_size 10 --checkpoint_model "models/TCI/Unet/U_net_pinv_N0_10_sig_0_N_64_M_512_epo_100_lr_0.001_sss_10_sdr_0.5_bs_64_reg_1e-07.pth"
```

```
python scripts/tci/train_EMConvNet_light.py --CR 512 --N0 10 --sig 0 --n_iter 4 --poisson_dc 1 --data_train_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --data_val_root "/gpfsscratch/rech/hbu/commun/stl-10/data" --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI/EM_net/" --expe_root "data/expe/" --num_epochs 41 --batch_size 64 --reg 0.000001 --step_size 10 --checkpoint_model "models/TCI/Unet/DCONV_c0mp_N0_10_sig_0_Denoi_N_64_M_512_epo_100_lr_0.001_sss_10_sdr_0.5_bs_64_reg_1e-07.pth"
```


##imagenet


```
python scripts/tci_imagenet/train_noisy_SDCAN_imagenet.py --CR 512 --N0 10 --sig 0 --model_recon 'U_net' --n_iter 0 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/imagenet" --data_val_root "/gpfsscratch/rech/hbu/commun/imagenet" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 0 --model_root "models/TCI_imagenet/Unet/" --expe_root "data/expe/" --num_epochs 50 --batch_size 64 --step_size 10
```

```
python scripts/tci_imagenet/train_noisy_SDCAN_conv_imagenet.py --CR 512 --N0 10 --sig 0 --n_iter 0 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/imagenet" --data_val_root "/gpfsscratch/rech/hbu/commun/imagenet" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI_imagenet/Unet/" --expe_root "data/expe/" --num_epochs 50 --batch_size 64 --step_size 10
```

```
python scripts/tci_imagenet/train_noisy_SDCAN_conv_imagenet.py --CR 512 --N0 10 --sig 0 --n_iter 0 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/imagenet" --data_val_root "/gpfsscratch/rech/hbu/commun/imagenet" --net_arch 0 --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI_imagenet/Unet/" --expe_root "data/expe/" --num_epochs 50 --batch_size 64 --step_size 10
```

```
python scripts/tci_imagenet/train_neumann_bis_imagenet.py --CR 512 --N0 10 --sig 0 --n_iter 4 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/imagenet" --data_val_root "/gpfsscratch/rech/hbu/commun/imagenet" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 0 --model_root "models/TCI_imagenet/neumann/" --expe_root "data/expe/" --num_epochs 20 --batch_size 64 --step_size 4 --checkpoint_model "models/TCI_imagenet/Unet/U_net_pinv_N0_10_sig_0_N_64_M_512_epo_50_lr_0.001_sss_10_sdr_0.5_bs_64_reg_1e-07.pth"
```

```
python scripts/tci_imagenet/train_modl_imagenet_2.py --CR 512 --N0 10 --sig 0 --n_iter 4 --poisson_dc 0 --data_train_root "/gpfsscratch/rech/hbu/commun/imagenet" --data_val_root "/gpfsscratch/rech/hbu/commun/imagenet" --net_arch 2 --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI_imagenet/modl/" --expe_root "data/expe/" --num_epochs 15 --batch_size 64 --step_size 4 --checkpoint_model "models/TCI_imagenet/Unet/U_net_pinv_N0_10_sig_0_N_64_M_512_epo_50_lr_0.001_sss_10_sdr_0.5_bs_64_reg_1e-07.pth"
```

```
python scripts/tci_imagenet/train_EMConvNet_net_imagenet_light.py --CR 512 --N0 10 --sig 0 --n_iter 4 --poisson_dc 1 --data_train_root "/gpfsscratch/rech/hbu/commun/imagenet" --data_val_root "/gpfsscratch/rech/hbu/commun/imagenet" --precompute_root "models/SDCAN/" --denoise 1 --model_root "models/TCI_imagenet/EM_net/" --expe_root "data/expe/" --num_epochs 16 --batch_size 64 --step_size 4 --checkpoint_model "models/TCI_imagenet/Unet/DCONV_c0mp_N0_10_sig_0_Denoi_N_64_M_512_epo_50_lr_0.001_sss_10_sdr_0.5_bs_64_reg_1e-07.pth"
```

