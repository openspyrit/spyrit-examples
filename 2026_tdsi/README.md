# Set-up

1. Install Ciotola's HS toolbox :
```
git clone https://github.com/matciotola/hyperspectral_pansharpening_toolbox.git
```

2. Intall dependencies of the toolbox in a new conda env: 
```
conda env create -n 2026_tdsi -f hyperspectral_pansharpening_toolbox/hs_pan_toolbox_environment.yaml
```

3. Activate the conda env:
```
conda activate 2026_tdsi
```

4. Install spyrit in editable mode:
```
git clone https://github.com/openspyrit/spyrit.git
cd spyrit
pip install -e .
```

5. Install pandas
```
pip install pandas
```

6. Install deepinverse
```
pip install deepinv
```



# Readme
This repo if for experimenting with different methods of the pansharpening state of the art, applied to single-pixel imaging. 

The single-pixel imaging physics is coded with spyrit.

The pansharpening methods are taken from the hyperspectral_pansharpening_toolbox, which is not always GPU compatible as is.

Pandas is used for loading the tristimulus response of the human color vision, from the CIE 1931 convention. 

Deepinv is used for the TV proximal operator.