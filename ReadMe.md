## Environment Configuration: 

- Python 3
- libraries:
  - import numpy as np
  - import os
  - import cv2
     * We need cv2 to read image data (it has nothing to do with the Neural Network structure itself)
     * pip install opencv-python
  - import pickle
     * Python’s built-in pickle module to serialize any Python object
  - import copy
     * to make a copy of the model, and to save a model during the training process as a checkpoint
  - Alternatively: import nnfs
     * to ensure repeatable results for training and testing
     * pip install nnfs

## Project Description: 
This project aims to build a neural network from scratch withouting using any AI libraries.

## Structures of the Neural Network:
- Layers.py: Definitions of various layer types
- Activation_Funcs.py: Definitions of various activation functions
- Optimizers.py: Definitions of various optimizers
- Losses.py: Definitions of various loss functions
- Accuracies: Definitions of accuracy calculations

- Model.py: Class definition for the neural network model.
  
  It contains functions that can: 
  * Add objects to the model
  * Set loss, optimizer and accuracy
  * Finalize the model
  * Train the model
  * Evaluates the model using passed-in dataset
  * Predicts on the samples
  * Performs forward and backword passes
  * Retrieves and returns parameters of trainable layers
  * Updates the model with new parameters
  * Saves the parameters to a file
  * Loads the weights and updates a model instance with them
  * Saves the model
  * Loads and returns a model

## Dataset preparation:
### This part is already done 
### and the dataset is in the "fashion_mnist_images" folder
- data_download.py: 
  * Run it to download The Fashion MNIST image dataset (or replace the link with other datasets)
  * The Fashion MNIST dataset is a collection of 60,000 training samples and 10,000 testing samples of 28x28 images of 10 various clothing items like shoes, boots, shirts, bags, and more.
  * We didn't use the regular MNIST dataset, which is a dataset of images of handwritten digits (0 through 9) at a resolution of 28x28 pixels, because it’s comically easy to get 99%+ accuracy.

- dataset_preparation.py: 
  * Run it to load and create a MNIST dataset for training and testing

## Model training:
### We train a model saved as "fashion_mnist.model"
- train_NN.py: 
  * To demenstrate, we constructed a model containing an input layer, 2 hidden layers using ReLU activation, and an output layer with softmax activation. Since we’re building a classification model, we used cross-entropy loss, Adam optimizer, and categorical accuracy
  * Run it to train and save a model 

## Test with Prediction:

- predict_on_dataset.py: 
  * we want to see if the model we trained actually works
  * run it to predict on the samples from validation dataset and print the result
  * compare the prediction to the true labels

- predict_on_images.py: 
  * run it to apply the model to predict on an images outside of our dataset

