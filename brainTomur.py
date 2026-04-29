# متاسفم . اما نشد دیتا ست را اپلود کنم . در اصرع وقت راهی پیدا میکنم
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
address = "brain tomour"

train = tf.keras.preprocessing.image_dataset_from_directory(
    address,
    validation_split = 0.2,
    seed=42,
    image_size=(64,64),
    batch_size=30,
    subset='training', #'validation'
    label_mode='categorical', #binary for clasification 2 classes
)

test = tf.keras.preprocessing.image_dataset_from_directory(
    address,
    validation_split = 0.2,
    seed=42,
    image_size=(64,64),
    batch_size=30,
    subset='validation', #'validation'
    label_mode='categorical', #binary for clasification 2 classes
)
model = Sequential()

model.add(Conv2D(32 , kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  
    metrics=['accuracy']
)
h = model.fit(train,epochs=7,validation_data=test)
model.save("brain.h5")
