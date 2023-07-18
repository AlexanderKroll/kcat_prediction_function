# Description
This repository contains an easy-to-use Python function for the kcat prediction model from our paper "Turnover number predictions for kinetically uncharacterized enzymes using machine and deep learning". 

## Predicting kcat values for enzyme-reaction pairs
The kcat prediction model was only trained with natural enzyme-reaction pairs with wild-type enzymes. Hence, the model will not be good at predicting kcat for mutants
or for non-natural reactions of enzymes. 

## Downloading data folder
Before you can run the kcat prediction function, you need to download and unzip a [data folder](https://doi.org/10.5281/zenodo.8038678). Afterwards, this repository should have the following strcuture:

    ├── code                   
    ├── data                    
    └── README.md

## substrate and product representations
You can use InChI strings, KEGG Compound IDs, and SMILES strings as substrate/product representations.

## Requirements

- python 3.7
- jupyter
- pandas 1.1.3
- torch 1.11.0
- numpy 
- rdkit 2020.09.1
- fair-esm 0.4.0
- py-xgboost 1.6.1

The listed packages can be installed using conda and anaconda:

```bash
pip install pandas==1.1.3
pip install torch==1.11.0
pip install numpy
pip install fair-esm==0.4.0
conda install -c conda-forge py-xgboost=1.6.1
conda install -c rdkit rdkit=2020.09.1
```

## Content

There is a Jupyter notebook "Tutorial kcat prediction.ipynb" in the folder "code" that contains an example on how to use the kcat prediction function.

## Problems/Questions
If you face any issues or problems, please open an issue.

