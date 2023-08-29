# Brain Tumor Classification with Keras and Hyperparameter Tuning

# Overview
This repository contains a Python script for classifying brain scan images into four categories: Glioma, Meningioma, No Tumor, and Pituitary. The classification is done using a Keras model based on the VGG16 architecture and hyperparameter optimization is achieved using Keras Tuner.

# Dataset
The dataset comprises approximately 7000 brain scan images.
Class Labels:
Glioma
Meningioma
No Tumor
Pituitary

# Requirements
Python 3.x
TensorFlow 2.x
Keras Tuner
NumPy

# Code Breakdown

Import Dependencies
The necessary modules are imported, including TensorFlow, Keras Tuner, and NumPy.

```from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from kerastuner.tuners import RandomSearch
import numpy as np```
