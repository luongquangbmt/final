#!/usr/bin/env python
# coding: utf-8

# In[47]:


#!pip install tensorflow
#!pip install opencv-python
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import tensorflow as tf
import tensorflow.keras.layers as tfl
import numpy as np
import os
import sys
import pandas as pd
 
import zipfile
with zipfile.ZipFile('TrainingData.zip', 'r') as zip_ref:
    zip_ref.extractall('TrainingData')
    

loading_args = dict(
    directory='TrainingData/Training Data',
    shuffle=True,
    batch_size=32,
    image_size=(64,64),
    validation_split=0.2,
    seed=0,
    label_mode='categorical',
    interpolation='area',
)
#calling image_dataset_from_directory(main_directory, labels='inferred') will return a tf.data.Dataset
#that yields batches of images from the subdirectories class_a and class_b, together with labels 0 and 1
train_dataset = image_dataset_from_directory(
    **loading_args,
    subset='training'
)

test_dataset = image_dataset_from_directory(
    **loading_args,
    subset='validation'
)


# In[60]:


class_names = train_dataset.class_names
#You can create an iterator object by applying the iter() built-in function to an iterable.
#use Python built-in next() function to get the next data element in the stream of data.
#From this, we are expecting to get a batch of samples.
first_batch = next(iter(train_dataset.take(1)))
images, labels = first_batch

plt.figure(figsize=(7, 7))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(images[i] / 255)
    ax.set_title(class_names[np.argmax(labels[i])])
    plt.axis('off')


# In[49]:


cnn = tf.keras.models.Sequential()


# In[50]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[51]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[52]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[53]:


cnn.add(tf.keras.layers.Flatten())


# In[54]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[55]:


cnn.add(tf.keras.layers.Dense(units=3, activation='sigmoid'))


# In[56]:


base_learning_rate = 0.001
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)


# In[57]:


cnn.fit(train_dataset,validation_data=test_dataset,epochs=25)


# In[127]:


from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from keras.preprocessing import image
import keras.utils as image

#A dataset that interleaves elements from datasets at random, 
#according to weights if provided, otherwise with uniform probability.
first_batch = tf.data.experimental.sample_from_datasets([test_dataset]) 
images, labels = next(iter(first_batch))
class_names = test_dataset.class_names

plt.figure(figsize=(7, 7))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(images[i] / 255)
    
    #A tensor with the same data as input, with an additional dimension inserted at the index specified by axis.
    pred = cnn.predict(tf.expand_dims(images[i], 0)).ravel()
    pred_class = np.argmax(pred)
    prob = "%.02f" % (pred[pred_class] * 100)
        
    ax.set_title(f'{class_names[pred_class]} ({prob}%)')
    plt.axis('off')


# In[74]:


cnn.summary()


# In[99]:


#Using ML models:

#first making an X and y variable
anime_path='TrainingData/Training Data/Anime/'
X = []
y = []
cnn_image_shape=(64,64)
for folder in os.scandir(anime_path):
    for file in os.scandir(anime_path + folder.name):
        img = cv2.imread(anime_path + folder.name + '/' + file.name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, cnn_image_shape)
        img = np.array(img, dtype='float32')
        X.append(img)
        y.append(0)
        
        
cartoon_path = 'TrainingData/Training Data/Cartoon/'

for folder in os.scandir(cartoon_path):
    for file in os.scandir(cartoon_path + folder.name):
        img = cv2.imread(cartoon_path + folder.name + '/' + file.name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, cnn_image_shape)
        img = np.array(img, dtype='float32')
        X.append(img)
        y.append(1)

X = np.array(X)
y = np.array(y)


# In[116]:


from sklearn.tree import DecisionTreeClassifier
#split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#reshape data since needs to be 2 dimensional
X_train=X_train.reshape(8002,64*64*3)
X_test=X_test.reshape(890,64*64*3)
from sklearn import tree

# Building Decision Tree model 
dtc = tree.DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)

y_predict = dtc.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
#find the accuracy of the prediction
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
accuracy_score(y_test, y_predict)


# In[117]:


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)
target_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
accuracy_score(y_test, y_predict)


# In[ ]:




