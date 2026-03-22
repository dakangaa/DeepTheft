# HMTheft

## Introduction
This repository provides the official implementation for the paper **"HMTheft: Layer-wise Hyperparameter Stealing Attack on DNNs with Unknown Inputs in Low-Altitude Wireless Networks"**. This project investigates methods for stealing layer-wise hyperparameters of Deep Neural Networks (DNNs) via side-channel information in Low-Altitude Wireless Network (LAWN) environments, specifically under scenarios where the target model's input data is unknown.

## Environment Setup
The project requires the following dependencies:
- **python** == 3.10
- **pandas** == 2.3.3
- **h5py** == 3.15.1
- **numpy** == 2.2.6
- **scikit-learn** == 1.7.2
- **torch** == 2.10.0
- **torchvision** == 0.25.0

## Data Preparation
This project utilizes the **RAPL-based dataset** from the paper *"DeepTheft: Stealing DNN Model Architectures through Power Side Channel"* by Gao et al.

The dataset is accessible via the original authors' link: [Zenodo Repository](https://zenodo.org/records/14334986).

**Note:** Please ensure that the raw HDF5 files (`data.h5` and `hp.h5`) are placed in the `dataset` directory.

## Running Guide

### 1. Data Preprocessing
Convert the data format to accelerate data loading speeds using the provided script:
```bash
python scripts/dataset_convert.py
```

### 2. Model Training
Train the model across multiple source domains. You can specify the layer type and target hyperparameter:
```bash
# Example: Predict the kernel size of a convolutional layer.
# Number of source domains: 4, Test domain: "331", Learning rate: 0.001.
python train.py --layer_type conv2d -H kernel_size -o 4 --test_domain 331 --lr 0.001
```

### 3. Evaluation
Test the model's performance on an unknown target domain:
```bash
# Example: Evaluate predictions for kernel size of a convolutional layer.
# Number of source domains: 4, Test domain: "331".
python test.py --layer_type conv2d -H kernel_size -o 4 --test_domain 331
```