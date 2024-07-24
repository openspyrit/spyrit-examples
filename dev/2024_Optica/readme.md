## Create a conda environment
```shell
conda create --name spyrit-dev
conda activate spyrit-dev
```

## Install spyrit in developper mode

### First, install pytorch using conda
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Then, clone spyrit and install it using pip
```shell
git clone  # commit ???
pip install -e .
```

## Missing to run the script
```shell
pip install ipykernel
pip install girder-client
```