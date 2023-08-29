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

# Import Dependencies
The necessary modules are imported, including TensorFlow, Keras Tuner, and NumPy.

```from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from kerastuner.tuners import RandomSearch
import numpy as np
```

# Define the Model Architecture
We define a function called build_model that creates the model architecture.

```def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('neurons', min_value=128, max_value=256, step=128), activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', values=['SGD', 'Adam']), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

# Data Augmentation
Data augmentation techniques are applied to the training dataset using ImageDataGenerator.

```train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
```

# Data Preparation
The training and validation data generators are configured by specifying the directory paths for the training and validation data.

```train_dir = 'Path/To/files/Training'
validation_dir = 'Path/To/files/Testing'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')
```

# Hyperparameter Tuning
Keras Tuner's RandomSearch is used for hyperparameter optimization.

```tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='cancertuner')
```

# Model Training
The model is trained using the .search method from Keras Tuner.

```tuner.search(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10)
```

# Results Summary
The best hyperparameters found during the search are displayed.

```print("Results Summary:")
tuner.results_summary()
```

# Model Prediction
The script also includes code to use the best model to predict the class of a given image.

```# Load an image file, resizing it to 150x150 pixels (the input shape for VGG16)
image_path = 'path_to_your_image.jpg'
image = load_img(image_path, target_size=(150, 150))

# Convert the image to a numpy array
image_array = img_to_array(image)

# Rescale the pixel values to the range [0, 1]
image_array = image_array / 255.0

# Expand the dimensions of the image array to include a batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Get the best model (as previously obtained)
best_model = tuner.get_best_models(num_models=1)[0]

# Predict the class of the image
predictions = best_model.predict(image_array)

# Get the index of the class with the highest probability
predicted_class = np.argmax(predictions)

# Mapping of class indices to class labels (as determined by the training data generator)
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Inverting the dictionary

# Get the label corresponding to the predicted class
predicted_label = class_labels[predicted_class]

print(f"The image is predicted to be: {predicted_label}")
```

# How to Run
1. Update train_dir and validation_dir with your training and validation data paths.
2. Run the script.
3. Examine the results summary to find the best hyperparameters.
4. The script will use the best model to predict the class label for the image at image_path.
Warning: the run time on the whole script maybe hours to days.
