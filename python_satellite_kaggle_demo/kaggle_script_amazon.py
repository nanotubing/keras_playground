# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook_preview.ipynb

# Imports
import glob
import numpy as np
import os#, sys
import pandas as pd
import matplotlib.pyplot as plt
# keras imports
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
#from sklearn.metrics import accuracy_score, f1_score
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

file_labels = []
file_labels2 = np.zeros((len(file_img_no), max_classes))
for i in file_img_no:
    temp = all_labels.iloc[i]['tags'].split(" ")
    print(temp)
    file_labels.append(temp)
del i

encoder = MultiLabelBinarizer()
file_labels2 = encoder.fit_transform(file_labels)
file_labels2 = np.asarray(file_labels2)

del all_labels

# Get image size
#train_image_size = np.asarray([train_img.shape[1], train_img.shape[2], train_img.shape[3]])
#test_image_size = np.asarray([test_img.shape[1], test_img.shape[2], test_img.shape[3]])

#probably not needed 20190226
#create training and validation sets based on an 80/20 split
split_size = 0.1
split_index = round(split_size * len(file_labels2))
#shuffled_indices = pd.DataFrame(np.random.permutation(train_labels))
#shuffled_indices.set_index(0, inplace = True)
#shuffled_indices.head()
#training_indices = shuffled_indices[0:split_index]
#test_indices = shuffled_indices[split_index:] 


#one-hot encode the labels
#def encode(data):
#    print('Shape of data (BEFORE encode): %s' % str(data.shape))
#    encoded = to_categorical(data)
#    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
#    return encoded
#file_labels_onehot = encode(file_labels)

## Split the images and the labels
x_test = file_img[split_index:, :, :]
y_test = file_labels2[split_index:]
x_train = file_img[0:split_index, :, :]
y_train = file_labels2[0:split_index]

##time to get down and dirty into the model
## Hyperparamater
N_LAYERS = 4

def cnn(size, n_layers):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = 20
    MAX_NEURONS = 120
    KERNEL = (3, 3)

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    nuerons = nuerons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            shape = (size[0], size[1], size[2])
            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
        else:
            model.add(Conv2D(nuerons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

## Instantiate the model
model = cnn(size=file_image_size, n_layers=N_LAYERS)

# Training hyperparamters
EPOCHS = 150
BATCH_SIZE = 40

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

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)
del log_dir, now

#load the model
#model = load_model('/Users/cschrader/Documents/GitHub/keras_playground/python_satellite_kaggle_demo/data/cschrader_model.h5')

#save the model
model.save('/Users/cschrader/Documents/GitHub/keras_playground/python_satellite_kaggle_demo/data/cschrader_model.h5')  

# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)
# Report the accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy: " + str(accuracy))
