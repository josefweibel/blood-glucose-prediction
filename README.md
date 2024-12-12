# Blood Glucose Prediction with Deep Learning

Corina Hüni, Isabella Torres, Joseph Weibel

## Installation

This code base requires Python 3.10. Dependencies are defined in `Pipfile` and can be installed using [pipenv](https://pipenv.pypa.io/) or any other package manager.

## Data

The orginal dataset [OhioT1DM](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881904/) must be put inside the `data` directory. It should contain directories `train` and `test` with the corresponding CSV files. Afterwards, run the preprocessing pipeline in `notebooks/prepare_data_pipeline.ipynb`. This will create the files `{train|val|test}_processed.csv` inside the `data` directory. These files are the basis for model training and evaluation.

## Notebooks

Other notebooks include code for data exploration, model training and evaluation of models. The most important files are `notebooks/model_training.ipynb` and `notebooks/model_evaluation.ipynb`. These notebook build up on a common code base stored in the `src` directory.

## Model Configurations

Every model has a unique name and is based on a configuration with that name. The configuration define the architecture, loss function, normalization, time series horizon and other hyperparameters of the models. All configurations are stored in the `config` directory.

## Results

All trained models are stored in the `models` directory along with their metadata (incl. loss values during training and scores on the validation set).

## Literature

Some important papers related to this work are stored in `papers`.