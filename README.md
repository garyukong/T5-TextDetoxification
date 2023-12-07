# Advanced Text Detoxification Techniques With T5 Models

## Overview

This repository hosts the code and data for the project "Advanced Text Detoxification Techniques With T5 Models: A Comparative Analysis Against BART". The project aims to enhance text detoxification using T5 models, incorporating advanced techniques such as bidirectional training, data augmentation, and negative lexically constrained decoding.

## Key Findings

- **T5 Model Performance**: T5 models, especially when enhanced with bidirectional training, data augmentation, and negative lexically constrained decoding, showed competitive or superior performance in text detoxification compared to BART models.
- **Impact of Advanced Techniques**: The integration of bidirectional training and data augmentation improved the style transfer accuracy and semantic integrity of the T5 models.
- **Effectiveness of NLCD**: Negative Lexically Constrained Decoding (NLCD) proved to be a pivotal technique in enhancing the T5 models' ability to maintain context while avoiding toxic expressions.

## Repository Structure

- `data/`: Contains raw, interim, and processed datasets used for training and evaluation.
- `docs/`: Documentation and related materials, including the project paper and proposal.
- `notebooks/`: Jupyter notebooks for data preprocessing, model training, and evaluation.
- `results/`: Results from model evaluation and hyperparameter tuning experiments.
- `requirements.txt`: List of Python packages required to run the code.

## Data Description

- `raw/`: Original, unmodified datasets mainly sourced from https://github.com/s-nlp/paradetox
- `interim/`: Intermediate data files used during processing.
- `processed/`: Final datasets used for model training and evaluation.

## Documentation

- `T5_TextDetoxification_Paper.pdf`: Detailed research paper.
- `T5_TextDetoxification_Proposal.gdoc`: Initial project proposal.

## Model Setup

For the Jupyter notebooks to run properly, it is essential to download the pre-trained models and save them in a `models/` directory within the project. Follow these steps:

1. **Download Models**:
   - Access the Google Drive folder containing the models: [T5-Small Detoxification Models](https://drive.google.com/drive/u/0/folders/1CmWYk0qtGmvLQsvnIJmNEDctkvs-0PDR).
   - Download the necessary model files to your local machine.

2. **Move Downloaded Models**:
   - Move or copy the downloaded model files into the `models/` directory.

After completing these steps, the Jupyter notebooks located in the `notebooks/` directory should be able to locate and utilize the models correctly.

## Notebooks

- `1_Data_Preprocessing.ipynb`: Cleans and converts raw data into Huggingface DatasetDict objects with appropriate train, validation and test splits
- `2_Data_Augmentation.ipynb`: Builds augmented datasets using back-translation
- `3_Model_Training.ipynb`: Trains variants of T5-Small models using Huggingface Trainer
- `4_Evaluation.ipynb`: Evaluates baseline and T5-Small models

## Installation

```bash
git clone https://github.com/garyukong/[YourRepoName].git
conda create --name [YourNewEnvName] --file requirements.txt
conda activate [YourNewEnvName]
```
