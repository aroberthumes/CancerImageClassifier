from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from kerastuner.tuners import RandomSearch
import numpy as np

def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('neurons', min_value=128, max_value=256, step=128), activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', values=['SGD', 'Adam']), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'Path/To/files/Training'
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

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

tuner.search(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10)

print("Results Summary:")
tuner.results_summary()

# Load an image file, resizing it to 150x150 pixels (the input shape for VGG16)
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
