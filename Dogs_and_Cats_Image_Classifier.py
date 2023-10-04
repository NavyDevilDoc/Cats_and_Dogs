import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from keras.models import load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Concatenate,
                                     Activation, ReLU, LeakyReLU)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import os
import shutil
from random import sample

'''
# This section is how I set up my Google Colab environment. This was my first time trying
# anything like this so if anyone knows how to do it better/more efficiently, don't hesitate
# to let me know.

# Mount a Google Drive 
from google.colab import drive
drive.mount('/content/gdrive')

# Copying training data
!mkdir -p /content/train/cats
!mkdir -p /content/train/dogs
!cp -r "/content/gdrive/MyDrive/Colab_Notebooks_2/PetImages/train/cats/"* /content/train/cats/
!cp -r "/content/gdrive/MyDrive/Colab_Notebooks_2/PetImages/train/dogs/"* /content/train/dogs/

# Copying test data
!mkdir -p /content/test/cats
!mkdir -p /content/test/dogs
!cp -r "/content/gdrive/MyDrive/Colab_Notebooks_2/PetImages/test/cats/"* /content/test/cats/
!cp -r "/content/gdrive/MyDrive/Colab_Notebooks_2/PetImages/test/dogs/"* /content/test/dogs/

# Creating validation folders and moving 400 random cat/dog images from test to validation
!mkdir -p /content/validation/cats
!mkdir -p /content/validation/dogs

for animal in ['cats', 'dogs']:
    src_folder = f'/content/test/{animal}/'
    dest_folder = f'/content/validation/{animal}/'

    images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for image in sample(images, 400):
        shutil.move(src_folder + image, dest_folder + image)
'''

# Convolutional block for the classifier
def conv_block(x, filters, kernel_size, pool_size=(2, 2), dropout_rate=0.2, use_dropout=True):
    x = Conv2D(filters, kernel_size, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size)(x)
    if use_dropout:
        x = Dropout(dropout_rate)(x)
    return x

# Set the data augmentation parameters and create the training, validation, and testing generators
# Note that if you're going to run this on your local machine, you'll need to change this line:
# '/content/train'
# to your file path
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    '/content/train',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    '/content/validation',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '/content/test',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary')

# Check to ensure the data splitting went off as expected. This assumes that cats
# are labeled as '0' and dogs are labeled as '1'. Generally speaking, I'm not the
# best at this but after multiple checks against what's actually in my file path, 
# everything was as it should be.
train_cats = sum(train_generator.labels == 0)  # Assuming 0 is the label for 'cat'
train_dogs = sum(train_generator.labels == 1)  # Assuming 1 is the label for 'dog'
print(f"Training set has {train_cats} cat images and {train_dogs} dog images.")

val_cats = sum(validation_generator.labels == 0)  
val_dogs = sum(validation_generator.labels == 1)  
print(f"Validation set has {val_cats} cat images and {val_dogs} dog images.")

test_cats = sum(test_generator.labels == 0)  
test_dogs = sum(test_generator.labels == 1)  
print(f"Test set has {test_cats} cat images and {test_dogs} dog images.")

'''
I went with a multi-headed CNN architecture with varying kernel sizes. Compared
to my initial model, which was a shallow CNN, test results increased from 45% to
97.7% at 100 epochs. Note that since I used early stopping, I never saw more than
85 epochs in a single run.
'''
input_layer = Input(shape=(200, 200, 3))

# First head
x1 = conv_block(input_layer, 64, (1, 1))
x1 = conv_block(x1, 128, (1, 1))
# Second head
x2 = conv_block(input_layer, 64, (3, 3))
x2 = conv_block(x2, 128, (3, 3))
# Third head
x3 = conv_block(input_layer, 64, (5, 5))
x3 = conv_block(x3, 128, (5, 5)) 
# Fourth head
x4 = conv_block(input_layer, 64, (7, 7))
x4 = conv_block(x4, 128, (7, 7))

# Concatenate the output of the four branches
merged = Concatenate()([x1, x2, x3, x4])

x = conv_block(merged, 256, (7, 7), use_dropout=False)
x = conv_block(x, 128, (5, 5), use_dropout=False)
x = conv_block(x, 64, (3, 3), use_dropout=False)
x = conv_block(x, 32, (1,1), use_dropout=True)

x = Flatten()(x)
x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
x = Dense(128, activation='relu', kernel_initializer='he_normal')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=20)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stop])

_, acc = model.evaluate_generator(test_generator,
                                  steps=len(test_generator),
                                  verbose=0)
print('Test Accuracy -> %.3f' % (acc * 100.0))

# Plot loss and accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Initialize empty arrays to store results
all_labels = []
all_predictions = []
all_images = []  # To store some images for later visualization

# Manually loop over the generator and make predictions
for i in range(len(test_generator)):
    images_batch, labels_batch = next(test_generator)
    predictions_batch = model.predict(images_batch)
    
    # Store some images, labels, and predictions
    all_labels.extend(labels_batch)
    all_predictions.extend(predictions_batch)
    
    if i == 0:  # Storing only the first batch of images
        all_images.extend(images_batch)

# Convert to NumPy arrays for easier slicing and indexing
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_images = np.array(all_images)  # Convert list of images to NumPy array

# Round predictions to nearest integer (0 or 1) to get the class labels
rounded_predictions = np.round(all_predictions).astype('int').flatten()

# Now your confusion matrix and classification reports will be aligned
print(confusion_matrix(all_labels, rounded_predictions))
print(classification_report(all_labels, rounded_predictions))

# Convert predictions and labels to binary labels (0: Cat, 1: Dog)
rounded_predictions_binary = np.where(all_predictions > 0.5, 1, 0).flatten()
labels_binary = all_labels.astype('int')

# Plot the first 10 images
for i in range(10):
    plt.figure(figsize=(8, 4))

    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(all_images[i])
    plt.title("Original Image")
    plt.axis('off')

    # Show actual and predicted labels
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f"Actual: {'Dog' if labels_binary[i] else 'Cat'}\nPredicted: {'Dog' if rounded_predictions_binary[i] else 'Cat'}",
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12)
    plt.axis('off')

    plt.show()
