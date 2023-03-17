# version 15 March 23
python ./train.py --stat_root data_online/ --data stl10 --num_epochs 1 --denoi cnn # just to check it runs
#python ./train.py --stat_root ../../stat/ILSVRC2012_v10102019 --M 4096 --img_size 128 --batch_size 256 --subs rect
#python ./train.py --stat_root ../../stat/ILSVRC2012_v10102019 --M 2048 --img_size 128 --batch_size 256
#python ./train.py --stat_root ../../stat/ILSVRC2012_v10102019 --M 1024 --img_size 128 --batch_size 256
#python ./train.py --stat_root ../../stat/ILSVRC2012_v10102019 --M 512 --img_size 128 --batch_size 256