from unet_model import *
from gen_patches import *

import os, sys, glob
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

N_BANDS = 4
N_CLASSES = 6  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3, 0.2]
#N_EPOCHS = 150
#N_EPOCHS = 50
N_EPOCHS = 3

UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 150
#TRAIN_SZ = 4000  # train size
TRAIN_SZ = 2000
VAL_SZ = 1000    # validation size
#image_path = './data/mband/{}.tif'
#mask_path = './data/gt_mband/{}.tif'
image_path = './data/planet_training/img/'
mask_path = './data/planet_training/mask/'

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

#trainIds = [str(i).zfill(2) for i in range(1, 25)]  # all availiable ids: from "01" to "24"
#trainIds = [ os.path.splitext(file)[0] for file in os.listdir(image_path)]
trainIds = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob('{}/*_MAT*.tif'.format(image_path))]

if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()
    
    overwrite_check = ['weights/unet_weights.hdf5', 'output/log_unet.csv', 'output/loss.png', 'output/accuracy.png']
    for file in overwrite_check:
        if os.path.exists(file):
            print('ERROR: file {0} already exists. Please rename or delete the following files before training your network:'.format(file))
            print(*overwrite_check, sep = '\n')
            sys.exit()

    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread(image_path+'{}.tif'.format(img_id)))
        mask = tiff.imread(mask_path+'{}_mask.tif'.format(img_id[:-11])) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')
    
    print("start train net")
    x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
    x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
    model = get_model()
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('output/log_unet.csv', append=True, separator=';')
    tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
    #change verbosity from 2 to 1
    model_fit_history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              verbose=1, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, tensorboard],
              validation_data=(x_val, y_val))
    
    #generate metrics and graphs
    #create a confusion matrix
#    y_pred = model.predict(x_val)
#    y_pred_reshape = y_pred.reshape(1000, -1)
#    y_val_reshape = y_val.reshape(1000, -1)
#    cm = confusion_matrix(y_val_reshape, y_pred_reshape)
#    print(cm)
#    f = open('confusion_matrix.txt', 'w')
#    f.write(cm)
#    f.close()
    metrics
    loss, acc = model.evaluate(x_val, y_val, verbose=0) #evaluate testing data and calculate loss and accuracy
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
    #plot training loss vs validation loss
    matplotlib.style.use('seaborn')
    epochs = len(model_fit_history.history['loss'])
    max_loss = max(max(model_fit_history.history['loss']), max(model_fit_history.history['val_loss']))
    plt.axis([0, epochs+1, 0, round(max_loss * 2.0) / 2 + 0.5])
    x = np.arange(1, epochs+1)
    plt.plot(x, model_fit_history.history['loss'])
    plt.plot(x, model_fit_history.history['val_loss'])
    plt.title('Training loss vs. Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='right')
    plt.savefig('output\\loss.png', bbox_inches='tight')
   #plot training accuracy vs validation accuracy
    matplotlib.style.use('seaborn')
    epochs = len(model_fit_history.history['acc'])
    plt.axis([0, epochs+1, 0, 1.2])
    x = np.arange(1, epochs+1)
    plt.plot(x, model_fit_history.history['acc'])
    plt.plot(x, model_fit_history.history['val_acc'])
    plt.title('Training accuracy vs. Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='right')
    plt.savefig('output\\accuracy.png', bbox_inches='tight')    
        