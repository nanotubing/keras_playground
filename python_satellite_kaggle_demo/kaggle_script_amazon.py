# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook_preview.ipynb

# Imports
import glob
import numpy as np
#import os.path as path
import os#, sys
import pandas as pd

#from scipy import misc
import matplotlib.pyplot as plt
# keras imports
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
#from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

# IMAGE_PATH should be the path to the downloaded amazon tiles in data folder
#this is necessary developing scripts interactively in the IDE
os.chdir(r"C:\Users\tuj53509\Documents\GitHub\keras_playground\data")
os.chdir("..\data")
print(os.getcwd())
IMAGE_PATH = str(os.getcwd()) + "\planet_amazon"
print(IMAGE_PATH)
os.listdir(IMAGE_PATH)

#generate a list of all tif images in the training directory
train_paths = glob.glob(os.path.join(IMAGE_PATH, "train-tif-v2", "*.tif"))
#read in each tif into a numpy array, and throw all of them in a big numpy array
#divide by 255 to rescale RGB values between 0 and 1. this allows the training
#to finish in our lifetimes
#only reading in the first 1000 images for memory reasons
#alternately, could use imagedatagenerator to train in batches
train_img = np.asarray([plt.imread(image)/255 for image in train_paths[:1000]])
del train_paths
#no longer necessary bc I moved it into command above
#train_img = np.asarray(train_img)

test_paths = glob.glob(os.path.join(IMAGE_PATH, "test-tif-v2", "*.tif"))
test_img = np.asarray([plt.imread(image)/255 for image in test_paths[:1000]])
del test_paths

# Get image size
train_image_size = np.asarray([train_img.shape[1], train_img.shape[2], train_img.shape[3]])
test_image_size = np.asarray([test_img.shape[1], test_img.shape[2], test_img.shape[3]])

#load labels
labels_df = pd.read_csv(r".\planet_amazon\train_v2.csv")
labels_df.head()

labels_df2 = []
for l in range(len(labels_df)):
    temp = labels_df.loc[l][1].split()
    temp.insert(0, l)
    labels_df2.append(temp)
labels_df3 = pd.DataFrame(labels_df2)
#labels_df3.set_index(0, inplace = True)
labels_df3.head()

#create training and validation sets based on an 80/20 split
split_size = 0.2
split_index = round(split_size * len(labels_df3.index))
shuffled_indices = pd.DataFrame(np.random.permutation(labels_df3))
shuffled_indices.set_index(0, inplace = True)
shuffled_indices.head()
training_indices = shuffled_indices[0:split_index]
test_indices = shuffled_indices[split_index:]


#
## Split the images and the labels
#x_train = images[train_indices, :, :]
#y_train = labels[train_indices]
#x_test = images[test_indices, :, :]
#y_test = labels[test_indices]
#
##define function to plot two rows of images
##one with airplanes, one without
#def visualize_data(positive_images, negative_images):
#    # INPUTS
#    # positive_images - Images where the label = 1 (True)
#    # negative_images - Images where the label = 0 (False)
#
#    figure = plt.figure()
#    count = 0
#    for i in range(positive_images.shape[0]):
#        count += 1
#        figure.add_subplot(2, positive_images.shape[0], count)
#        plt.imshow(positive_images[i, :, :])
#        plt.axis('off')
#        plt.title("1")
#
#        figure.add_subplot(1, negative_images.shape[0], count)
#        plt.imshow(negative_images[i, :, :])
#        plt.axis('off')
#        plt.title("0")
#    plt.show()
#
## Number of positive and negative examples to show
#N_TO_VISUALIZE = 10
#
## Select the first N positive examples
#positive_example_indices = (y_train == 1)
#positive_examples = x_train[positive_example_indices, :, :]
#positive_examples = positive_examples[0:N_TO_VISUALIZE, :, :]
#
## Select the first N negative examples
#negative_example_indices = (y_train == 0)
#negative_examples = x_train[negative_example_indices, :, :]
#negative_examples = negative_examples[0:N_TO_VISUALIZE, :, :]
#
## Call the visualization function
#visualize_data(positive_examples, negative_examples)
#
#
##time to get down and dirty into the model
## Hyperparamater
#N_LAYERS = 4
#
#def cnn(size, n_layers):
#    # INPUTS
#    # size     - size of the input images
#    # n_layers - number of layers
#    # OUTPUTS
#    # model    - compiled CNN
#
#    # Define hyperparamters
#    MIN_NEURONS = 20
#    MAX_NEURONS = 120
#    KERNEL = (3, 3)
#
#    # Determine the # of neurons in each convolutional layer
#    steps = np.floor(MAX_NEURONS / (n_layers + 1))
#    nuerons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
#    nuerons = nuerons.astype(np.int32)
#
#    # Define a model
#    model = Sequential()
#
#    # Add convolutional layers
#    for i in range(0, n_layers):
#        if i == 0:
#            shape = (size[0], size[1], size[2])
#            model.add(Conv2D(nuerons[i], KERNEL, input_shape=shape))
#        else:
#            model.add(Conv2D(nuerons[i], KERNEL))
#
#        model.add(Activation('relu'))
#
#    # Add max pooling layer
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Flatten())
#    model.add(Dense(MAX_NEURONS))
#    model.add(Activation('relu'))
#
#    # Add output layer
#    model.add(Dense(1))
#    model.add(Activation('sigmoid'))
#
#    # Compile the model
#    model.compile(loss='binary_crossentropy',
#                  optimizer='adam',
#                  metrics=['accuracy'])
#
#    # Print a summary of the model
#    model.summary()
#
#    return model
#
## Instantiate the model
#model = cnn(size=image_size, n_layers=N_LAYERS)
#
## Training hyperparamters
#EPOCHS = 150
#BATCH_SIZE = 200
#
## Early stopping callback
#PATIENCE = 10
#early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
#
## TensorBoard callback
#LOG_DIRECTORY_ROOT = '.'
#now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
#tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)
#
## Place the callbacks in a list
#callbacks = [early_stopping, tensorboard]
#
## Train the model
#model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)
#
##load the model
##model = load_model('/Users/cschrader/Documents/GitHub/keras_playground/python_satellite_kaggle_demo/data/cschrader_model.h5')
#
##save the model
#model.save('/Users/cschrader/Documents/GitHub/keras_playground/python_satellite_kaggle_demo/data/cschrader_model.h5')  
#
## Make a prediction on the test set
#test_predictions = model.predict(x_test)
#test_predictions = np.round(test_predictions)
## Report the accuracy
#accuracy = accuracy_score(y_test, test_predictions)
#print("Accuracy: " + str(accuracy))
#
#
#def visualize_incorrect_labels(x_data, y_real, y_predicted):
#    # INPUTS
#    # x_data      - images
#    # y_data      - ground truth labels
#    # y_predicted - predicted label
#    count = 0
#    figure = plt.figure()
#    incorrect_label_indices = (y_real != y_predicted)
#    y_real = y_real[incorrect_label_indices]
#    y_predicted = y_predicted[incorrect_label_indices]
#    x_data = x_data[incorrect_label_indices, :, :, :]
#
#    maximum_square = np.ceil(np.sqrt(x_data.shape[0]))
#
#    for i in range(x_data.shape[0]):
#        count += 1
#        figure.add_subplot(maximum_square, maximum_square, count)
#        plt.imshow(x_data[i, :, :, :])
#        plt.axis('off')
#        plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize=10)
#
#    plt.show()
#
#visualize_incorrect_labels(x_test, y_test, np.asarray(test_predictions).ravel())
