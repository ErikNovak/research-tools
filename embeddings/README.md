# Word Embeddings
This repository contains the jupyter notebook presentations about the word embeddings and language models (i.e. BERT).

## Prerequisites

- Python version 3.8 or higher (we will create a new environment with `Conda`)

  To test that your python version is correct, run `python --version` in the command line


## Creating the Conda environment
To create a new conda environment, one must first have `conda` enabled in their terminal. This can be done by installing either Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once done, run the following commands in the terminal:
```bash
# create the environment with the required packages
conda env create -f environment.yml
# to activate the conda environment run
conda activate embeddings
```

To deactivate the environment run:
```bash
conda deactivate
```

### Updating the conda environment

To update the environment with the new/removed packages run:
```bash
# 1st version: by activating the environment
conda activate embeddings
conda env update -f environment.yml
# 2nd version: without activating the environment
conda env update --name embeddings -f environment.yml
```

To update the `enviroment.yml` file with new packages run:
```bash
conda env export | grep -v "^prefix: " > environment.yml
```

## Jupyter Notebooks
To access and modify the notebooks in the `notebooks` folder run the following command:
```bash
jupyter notebook
```

## Jupyter Notebooks Conversion
To convert the jupyter notebooks to a presentation run the following command:
```bash
jupyter nbconvert --config mycfg.py
```
**NOTE:** The `mycfg.py` script contains the paths of the notebooks to be converted. If other notebooks need to be converted, appropriately modify the mentioned script.


## Folder structure

| Folder        | Description                                                                |
| ------------- | -------------------------------------------------------------------------- |
| models        | The folder containing the pytorch implementations of various models        |
| notebooks     | The folder containing the notebooks (used for explaining concepts)         |
| presentations | The folder containing the PDF presentations of notebooks                   |
