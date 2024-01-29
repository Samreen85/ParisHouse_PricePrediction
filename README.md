# House Price Prediction Model
## Overview:

This repository contains a machine learning model for predicting house prices using the Random Forest Regressor algorithm. The model is built in Python and utilizes the scikit-learn library for machine learning tasks. The dataset used for training and testing is sourced from ParisHousing.csv.
## Prerequisites:

Ensure you have the necessary dependencies installed before running the model. The key libraries required are:

    * scikit-learn: A machine learning library that provides simple and efficient tools for data analysis and modeling.
    * pandas: A data manipulation and analysis library that provides data structures for efficiently storing large datasets.

You can install these libraries using the package manager pip:pip install scikit-learn pandas

## Dataset:

The dataset (ParisHousing.csv) is a crucial component of this project. It is loaded into the model to facilitate the training and testing processes. The initial step of the script involves displaying the first 5 rows of the dataset to provide a glimpse of the available data.
## Pre-processing:

A pre-processing step is conducted to identify and handle any null values in the dataset. The script prints a message indicating the absence of null values. In case null values are detected, further pre-processing steps may be required based on the specific characteristics of the dataset.
## Model Training and Testing:

The dataset is divided into training and testing sets using the scikit-learn library's train_test_split function. The Random Forest Regressor model is then trained on the training set. Subsequently, the model's performance is evaluated on the testing set. The evaluation metrics employed include Mean Squared Error (MSE) and R^2 Score.
## Usage:

To run the model, follow these steps:
1) Clone the repository:
* git clone https://github.com/your-username/your-repo.git
  
2) Navigate to the project directory:
* cd your-repo

3) Run the Python script:
* python your_script.py
