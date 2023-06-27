# Ship Segmentation Project

This project focuses on ship segmentation in images using a dataset of annotated ship images. The goal is to develop a model that can accurately identify and segment ships in images.

## Dataset

The dataset for this project consists of ship images and corresponding segmentation information. The images are stored in the `data/train_v2` directory, and the segmentation annotations are provided in the `train_ship_segmentations_v2.csv` file. Each annotation includes the encoded pixels representing the ship's segmentation mask.

## Exploratory Data Analysis

The `exploratory data analysis.ipynb` notebook contains an analysis of the dataset to gain insights and understand the data distribution. It performs the following tasks:

- Reads and preprocesses the ship segmentation data.
- Extracts features from the images and adds them to the dataset.
- Identifies and removes any corrupted images.
- Explores the data by analyzing image dimensions, ship counts, and distributions.
- Displays ship segmentation pixels for sample images.

## Model Training

The `Model.py` file contains the code for creating the ship segmentation model. It defines the architecture and provides functions for training and saving the model checkpoints.

The `create_test_generator` function in the `OtherFile.py` file is used to create a data generator for generating predictions on the test set.

The `TestModel.py` script demonstrates how to load the trained model weights, read the test data, and generate predictions using the model.

## Usage

To use this project, follow these steps:

1. Set up the Python environment with the required dependencies. You can use the `requirements.txt` file to install the necessary packages.
2. Run the `exploratory data analysis.ipynb` notebook to explore the dataset and gain insights.
3. Train the ship segmentation model using the `Model.py` file. Adjust the hyperparameters and model architecture as needed.
4. Use the trained model to generate predictions on the test data by running the `TestModel.py` script.





