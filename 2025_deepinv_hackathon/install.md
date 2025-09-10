# windows with conda and spyder 6.0.7

conda create --name spyrit_deepinv_dev
conda activate spyrit_deepinv_dev
conda install pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126


git clone ...
cd [path to spyrit folder]
pip install -e .

git clone https://github.com/deepinv/deepinv.git
cd [path to deepinv folder]
pip install -e .

pip install spyder-kernels==3.0.*