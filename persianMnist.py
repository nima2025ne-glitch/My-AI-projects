# ببخشید . متاسفانه نتونستم دیتاست رو اپلود کنم . به زودی فکری براش میکنم 

import tensorflow as tf 
from keras import layers, models
import numpy as np
address = 'dataset'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    address,
    validation_split=0.2,
    seed=42,
    image_size=(16, 64),
    batch_size=30,
    subset='training',
    label_mode='categorical',
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    address,
    validation_split=0.2,
    seed=42,
    image_size=(16, 64),
    batch_size=30,
    subset='validation',
    label_mode='categorical',
)
train_images = []
train_labels = []
for images, labels in train_ds:
    train_images.append(images)
    train_labels.append(labels)
train_images = tf.image.rgb_to_grayscale(tf.concat(train_images, axis=0))
train_labels = tf.concat(train_labels, axis=0)

test_images = []
test_labels = []
for images, labels in test_ds:
    test_images.append(images)
    test_labels.append(labels)
test_images = tf.image.rgb_to_grayscale(tf.concat(test_images, axis=0))
test_labels = tf.concat(test_labels, axis=0)

train_images = train_images.numpy()
train_labels = train_labels.numpy()
test_images = test_images.numpy()
test_labels = test_labels.numpy()

print("Original image batch shape:", train_images.shape)
print("Grayscale image batch shape:", train_images.shape)
print("Label batch shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)
model = models.Sequential()

# Adjusted input_shape to match the new image size and grayscale (1 channel)
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(16, 64, 1)))
model.add(layers.MaxPooling2D((2, 2))) # Reduces to (32, 64)
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2))) # Reduces to (16, 32)
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2))) # Reduces to (8, 16) - Safely allows next Conv2D
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) # Assuming 10 classes based on the previous code

model.summary()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), # Use CategoricalCrossentropy for one-hot encoded labels
    metrics=['accuracy']
)
h = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images,test_labels))
model.save("model.keras")
# loaded_model = tf.keras.models.load_model('my_saved_model')
