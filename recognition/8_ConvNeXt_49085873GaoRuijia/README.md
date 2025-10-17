# Classifier for ADNI brain data based on the ConvNeXt

**Table of Contents**
  - [Model and Problem Description](#model-and-problem-description)
  - [Model Architecture](#model-architecture)
  - [About the Dataset](#about-the-dataset)
  - [Usage](#usage)
  - [Results](#results)
  - [References](#references)

## Model and Problem Description



## Model Architecture

![ConvNeXt Architecture](images/convnext_arch.png)

## About the Dataset



## Usage 
### Files Description
1. modules.py containing the source code of the components of ConvNeXt model. 
2. dataset.py containing the data loader for loading and preprocessing ANDI dataset. 
3. train.py containing the source code for training, validating, testing and saving the model. 
4. predict.py showing example usage of the trained model. 
5. README.md to sufficiently document the project.
6. images/ containing figures and files used in README.
   
### Dependencies

| Dependencies | Version | 
| :-----: | :-----: | 
| Python       | 3.9 |  
| torchvision  | 0.22.1+cu118 | 
| torch        | 2.7.1+cu118 | 
| numpy        | 2.1.2 | 
| scikit-learn | 1.7.1 | 
| matplotlib   | 3.10.5 | 
| pillow       | 11.0.0 | 
| tqdm         | 4.67.1 | 

### Dataset Structure

AD_NC
├── train
│   ├── AD
│   │   ├── 218391_78.jpeg
│   │   ├── ...│
│   └── NC
│       ├── 808819_88.jpeg
│       ├── ...│
└── validation
    ├── AD
    │   ├── 388206_78.jpeg
    │   ├── ...│
    └── NC
        ├── 1182968_94.jpeg
        ├── ...

### Training
### Predictions

## Results



## References



