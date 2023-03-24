 # M = 512 measurements (default)
 python train.py
 python train.py --no_denoi
 python train.py --no_denoi --net_arch 3
 python train.py --no_denoi --intensity_max 'inf'
 
 
 # M = 1024 measurements
 python train.py --CR 1024
 python train.py --CR 1024 --no_denoi 
 python train.py --CR 1024 --no_denoi --net_arch 3
 python train.py --CR 1024 --no_denoi --intensity_max 'inf'