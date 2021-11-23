 # M = 512 measurements (default)
 python train.py
 python train.py --denoi 0
 python train.py --denoi 0 --net_arch 3
 python train.py --denoi 0 --intensity_max None
 
 
 # M = 1024 measurements
 python train.py --CR 1024
 python train.py --denoi 0 --CR 1024
 python train.py --denoi 0 --net_arch 3 --CR 1024
 python train.py --denoi 0 --intensity_max None --CR 1024