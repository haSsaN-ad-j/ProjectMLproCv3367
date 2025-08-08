import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator




train_dir = 'C:/Users/Admin/Downloads/archive/chest_xray/train'
val_dir = 'C:/Users/Admin/Downloads/archive/chest_xray/val'
test_dir = 'C:/Users/Admin/Downloads/archive/chest_xray/test'



img_height, img_width = 150, 150
batch_size = 32

augment_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = augment_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,  
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

sample_normal = os.path.join(train_dir, 'NORMAL', os.listdir(os.path.join(train_dir, 'NORMAL'))[0])
sample_pneumonia = os.path.join(train_dir, 'PNEUMONIA', os.listdir(os.path.join(train_dir, 'PNEUMONIA'))[0])

img = Image.open(sample_normal)
plt.imshow(img)
plt.title('Sample Normal X-ray')
plt.show()

img = Image.open(sample_pneumonia)
plt.imshow(img)
plt.title('Sample Pneumonia X-ray')
plt.show()




model = tf.keras.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())  
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid')) 

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics = ['accuracy']
)

validation = model.fit(train_generator, epochs=10, validation_data=val_generator)
print(validation)


test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.4f}")

ytrue = test_generator.classes
y_probs = model.predict(test_generator)
y_test = (y_probs > 0.5).astype("int32").flatten()
print(confusion_matrix(ytrue,y_test))
print(classification_report(ytrue, y_test, target_names=test_generator.class_indices.keys()))



