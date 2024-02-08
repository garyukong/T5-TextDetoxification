# Advanced Text Detoxification Techniques With T5 Models

## Introduction

The prevalence of toxic comments on online platforms has escalated the need for sophisticated text detoxification methods. This project introduces an innovative approach using T5 models, titled "Advanced Text Detoxification Techniques With T5 Models: A Comparative Analysis Against BART". It focuses on enhancing the capabilities of T5 models through bidirectional training, data augmentation, and negative lexically constrained decoding (NLCD), aiming to set new benchmarks in text detoxification.

## Goals

- To evaluate the effectiveness of T5 models, enhanced through advanced techniques, for text detoxification tasks.
- To conduct a comparative analysis of T5 models against BART models, assessing their performance in maintaining style transfer accuracy and semantic integrity.
- To explore the contributions of bidirectional training, data augmentation, and NLCD to the field of text detoxification.

## Dataset

The project utilizes the ParaDetox dataset, a comprehensive collection designed for the detoxification of text, encompassing a wide range of toxic comments for model evaluation.

### Source:
[ParaDetox Project](https://github.com/s-nlp/paradetox)

## Methodology

1. **Data Preprocessing**: Initial steps involve cleaning and preparing the raw data for the training process.
2. **Data Augmentation**: Techniques like back-translation are used to enrich the training dataset, enhancing model robustness.
3. **Model Training**: Employs advanced T5 models, comparing them directly with BART models to gauge performance improvements.
4. **Evaluation**: Utilizes a suite of metrics to assess and compare the detoxification effectiveness of the models.

## Results

The study reveals that T5 models, particularly when augmented with the proposed advanced techniques, outperform BART models in text detoxification. The implementation of NLCD stands out as a significant enhancement, enabling the models to better navigate the complexities of toxic language mitigation while preserving content fidelity.

## Usage

To leverage this work for your purposes:

1. **Clone the Repository**: Access the code and resources by cloning the GitHub repository.
2. **Install Dependencies**: Set up your environment by installing the necessary Python packages listed in `requirements.txt`.
3. **Download Pre-trained Models**: Ensure the models are available locally by downloading them from the provided Google Drive link and placing them in the `models/` directory.

- Models Download Link: [T5-Small Detoxification Models on Google Drive](https://drive.google.com/drive/u/0/folders/1CmWYk0qtGmvLQsvnIJmNEDctkvs-0PDR)

4. **Run the Notebooks**: Execute the Jupyter notebooks within the `notebooks/` directory to replicate the data processing, model training, and evaluation steps.

## Future Work

Potential future directions include investigating the integration of T5 models with additional linguistic features and datasets to further refine detoxification strategies. The nuanced impact of NLCD across diverse toxic content types also presents an intriguing area for deeper exploration.

Project Organization
------------

    ├── README.md                                                      
    ├── data
    │   ├── processed                                                  <- Processed datasets ready for model training.
    │   ├── raw                                                        <- Original, unmodified datasets.
    │   └── interim                                                    <- Data at intermediate processing stages.
    │
    ├── docs                                                           <- Documentation including project papers and proposals.
    │
    ├── models                                                         <- Pre-trained models directory.
    │
    ├── notebooks                                                      <- Jupyter notebooks for preprocessing, training, and evaluation.
    |   ├── 1_Data_Preprocessing.ipynb                                 
    │   ├── 2_Data_Augmentation.ipynb                                  
    │   └── 3_Model_Training.ipynb                                     
    │   └── 4_Evaluation.ipynb                                         
    │
    ├── results                                                        <- Stored results from model evaluations.
    │
    └── requirements.txt                                               <- Python package requirements for the project.
