import tensorflow as tf 
from keras import datasets, layers, models
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images.shape
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
class_names[train_labels[5070][0]]
plt.close()
plt.imshow(train_images[2])
plt.show()
model = models.Sequential()

model.add(layers.Conv2D(32 , kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
model.compile(
    optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
h = model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))
plt.close()
plt.imshow(test_images[3])
plt.show()
print(class_names[test_labels[3][0]])
out = model.predict(test_images)
out[3]
m=-1000
o2=out[1650]
ind=-1

for i in range(len(o2)):
    if o2[i]>m:
        m=o2[i]
        ind=i


print(class_names[ind])
import cv2
img=cv2.imread('j4/asb.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img.shape
img=cv2.resize(img,(32,32))
img.shape
plt.close()
plt.imshow(img)
plt.show()
import numpy as np
img=np.array([img])
img.shape
output = model.predict(img)
m=-1000
o2=output[0]
ind=-1

for i in range(len(o2)):
    if o2[i]>m:
        m=o2[i]
        ind=i


print(class_names[ind])
