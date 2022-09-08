### just test if it runs
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --denoi cnnbn
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --subs rect
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 1 --subs rand

### pinv net | CNN | stl 10
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 4095 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 2018 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 1024 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 512 --arch pinv-net
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 256 --arch pinv-net

### pinv net | U-Net | stl 10
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 4095 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 2018 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 1024 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 512 --arch pinv-net --denoi unet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --data stl10 --M 256 --arch pinv-net --denoi unet

### pinv net | U-Net | ImageNet
#python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --M 4095
python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --M 2018
python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --M 1024
python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --M 512
python ./train.py --stat_root data_online/ --model_root model_v2/ --num_epochs 30 --M 256