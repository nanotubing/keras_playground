# -*- coding: utf-8 -*-
"""
script building a neural network to create a land cover classification map
"""
#https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook_preview.ipynb
#https://github.com/ZFTurbo/Kaggle-Planet-Understanding-the-Amazon-from-Space
#https://medium.com/@kylepob61392/airplane-image-classification-using-a-keras-cnn-22be506fdb53
#https://github.com/tavgreen/landuse_classification
#https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

# Imports
import glob
import numpy as np
import os#, sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
# keras imports
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime

# IMAGE_PATH should be the path to the downloaded amazon tiles in data folder
#this is necessary developing scripts interactively in the IDE
os.chdir(r"C:\Users\tuj53509\Documents\GitHub\keras_playground\data")
#os.chdir("..\data")
print(os.getcwd())
IMAGE_PATH = str(os.getcwd()) + "\planet_amazon"
print(IMAGE_PATH)
os.listdir(IMAGE_PATH)

#generate a list of all tif images in the training directory
file_paths = glob.glob(os.path.join(IMAGE_PATH, "train-tif-v2", "*.tif"))
file_paths_subset = file_paths[:1000]
del file_paths
#read in each tif into a numpy array, and throw all of them in a big numpy array
#divide by 255 to rescale RGB values between 0 and 1. this allows the training
#to finish in our lifetimes
#only reading in the first 1000 images for memory reasons
#alternately, could use imagedatagenerator to train in batches
all_labels = pd.read_csv(r".\planet_amazon\train_v2.csv")

file_img_no = []
for i in file_paths_subset:
    file_img_no.append(int(os.path.splitext(os.path.split(i)[1])[0].split("_")[1]))
del i

file_img = []
file_img = np.asarray([plt.imread(image)/255 for image in file_paths_subset])
file_image_size = np.asarray([file_img.shape[1], file_img.shape[2], file_img.shape[3]])
print(file_image_size)

#calculate number of classes in labels
max_classes = 0
for i in file_img_no:
    if len(all_labels.iloc[i]['tags'][:].split()) > max_classes:
        max_classes = len(all_labels.iloc[i]['tags'])
del i

file_labels = []
for i in file_img_no:
    temp = all_labels.iloc[i]['tags'].split(" ")
    print(temp)
    file_labels.append(temp)
del i

encoder = MultiLabelBinarizer()
file_labels2 = encoder.fit_transform(file_labels)
file_labels2 = np.asarray(file_labels2)

del all_labels

#create training and validation sets based on an 80/20 split
split_size = 0.9
split_index = round(split_size * len(file_labels2))

## Split the images and the labels
x_test = file_img[split_index:, :, :]
y_test = file_labels2[split_index:]
x_train = file_img[0:split_index, :, :]
y_train = file_labels2[0:split_index]

##time to get down and dirty into the model
## Hyperparamater
N_LAYERS = 4

model = Sequential() #model = sequential 
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=file_image_size)) #layer convolutional 2D
model.add(MaxPooling2D(pool_size=(2,2))) #max pooling with stride (2,2)
model.add(Conv2D(32, (3, 3), activation='relu')) #layer convolutional 2D
model.add(MaxPooling2D(pool_size=(2,2))) #max pooling with stride (2,2)
model.add(Dropout(0.25)) #delete neuron randomly while training and remain 75%
model.add(Flatten()) #make layer flatten
model.add(Dense(128, activation='relu')) #fully connected layer
model.add(Dropout(0.5)) #delete neuron randomly and remain 50%
model.add(Dense(17, activation='softmax')) #softmax works
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #setting loss function and optimizer
model.summary()

# Training hyperparamters
EPOCHS = 100
BATCH_SIZE = 250
# Early stopping callback
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = '.'
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test)) #training with epochs 100, batch size = 50
loss, acc = model.evaluate(x_test, y_test, verbose=0) #evaluate testing data and calculate loss and accuracy
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

#del log_dir, now

#load the model
#model = load_model('/Users/cschrader/Documents/GitHub/keras_playground/python_satellite_kaggle_demo/data/cschrader_model.h5')
#save the model
model.save('cschrader_model_20190311.h5')  

# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)
# Report the accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))
