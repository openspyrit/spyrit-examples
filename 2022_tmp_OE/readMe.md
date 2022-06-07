# Optics Express paper, first test using pip spyrit

*Authors:* N Ducros, G. Beneti-Martin, L Mahieu-Williame

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

*Authors:* N Ducros, A Lorente Mur, G. Beneti-Martin, L Mahieu-Williame

*Contact:* nicolas.ducros@insa-lyon.fr, CREATIS Laboratory, University of Lyon, France.

## Install the dependencies

1. We recommend creating a virtual (e.g., conda) environment first.

   ```shell
   # conda (or pip) install
   conda create --name new-env
   conda activate new-env
   conda install -c anaconda spydery
   conda install -c conda-forge jupyterlab
   conda install -c anaconda scikit-image
   conda install -c anaconda h5py 
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # for windows
   ```

   Alternatively, you can clone an existing environment with `conda create --name new-env --clone existing-env`

2. Our scripts primarily relies on the [SPyRiT ](https://github.com/openspyrit/spyrit) package that can be installed via `pip`.  We also need the spas package (see openspyrit/spas/) to get access to the metadata.

   NB: On Windows, you need to install [torch](https://pytorch.org/get-started/locally/) before SPyRiT

   ```shell
   # pip install
   pip install spyrit # tested with spyrit==1.2.0
   pip install spas   # tested with spas== ??1.1.0
   ```


## Get the scripts, networks and raw data

1.  Get source code and navigate to the `/2022_tmp_OE/` folder

   ```shell
   git clone https://github.com/openspyrit/spyrit-examples.git
   cd spyrit-examples/2022_OE/ 
   ```

2. Download the trained EM-Net model and raw data at this [url](https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2022_tmp_OE/2022_tmp_OE.zip) and extract its content in `models`

* Windows PowerShell

```powershell
wget https://www.creatis.insa-lyon.fr/~ducros/spyritexamples/2022_tmp_OE/2022_tmp_OE.zip -outfile data.zip
tar xvf data.zip 
```

The directory structure should be

```
|---spyrit-examples
|   |---2022_tmp_OE
|   |   |---data
|   |   |   |---
|   |   |---models
|   |   |   |---
|   |   |---stats
|   |   |   |---
|   |   |---analysis.py
|   |   |---
```
