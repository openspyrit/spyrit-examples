### just test if it runs
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --denoi cnnbn
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --subs rect
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --subs rand

### dc-net | U-Net | stl 10
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 2048 --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2 --data stl10 --M 2048 --denoi unet

### pinv net | CNN | stl 10
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 4095 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 2048 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 1024 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 512 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 256 --arch pinv-net

python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 4095 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 2048 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 1024 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 512 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 256 --arch pinv-net

python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 4095 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 2048 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 1024 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 512 --arch pinv-net
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 256 --arch pinv-net

### pinv net | U-Net | stl 10
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 4095 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 2048 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 1024 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 512 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 256 --arch pinv-net --denoi unet

python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 4095 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 2048 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 1024 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 512 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 50 --data stl10 --M 256 --arch pinv-net --denoi unet

python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 4095 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 2048 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 1024 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 512 --arch pinv-net --denoi unet
python ./train.py --stat_root data_online/ --model_root model_v2/ --N0 2500 --data stl10 --M 256 --arch pinv-net --denoi unet

### pinv net | CNN batch norm | stl 10
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 4095 --denoi cnnbn --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 2048 --denoi cnnbn --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 1024 --denoi cnnbn --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 512 --denoi cnnbn --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 256 --denoi cnnbn --arch pinv-net

### dc net | CNN batch norm | stl 10
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 4095 --denoi cnnbn
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 2048 --denoi cnnbn
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 1024 --denoi cnnbn
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 512 --denoi cnnbn
#python ./train.py --stat_root data_online/ --model_root model_v2/ --data stl10 --M 256 --denoi cnnbn

### dc net | CNN | ImageNet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --M 4095
#python ./train.py --stat_root data_online/ --model_root model_v2/ --M 2048
#python ./train.py --stat_root data_online/ --model_root model_v2/ --M 1024
#python ./train.py --stat_root data_online/ --model_root model_v2/ --M 512
#python ./train.py --stat_root data_online/ --model_root model_v2/ --M 256

### dc net | CNN | ImageNet | 128
#python ./train.py --img_size 128 --stat_root ../../stat/ILSVRC2012_v10102019 --model_root model_v2/  --M 1024 --batch_size 512
#python ./train.py --img_size 128 --stat_root ../../stat/ILSVRC2012_v10102019 --model_root model_v2/  --M 512 --batch_size 512