# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:15:30 2019
"""

import os, sys, glob
import numpy as np
import tifffile as tiff

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

image_path = './data/planet_training/img/'
mask_path = './data/planet_training/mask/'

trainIds = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob('{}/*_MAT*.tif'.format(image_path))]

if __name__ == '__main__':
 
    print('Reading images')
    for img_id in trainIds:
        mask_id = '{}_mask'.format(img_id[:-11]) 
        img_m = normalize(tiff.imread(image_path+'{}.tif'.format(img_id)))
        mask = tiff.imread(mask_path+'{}.tif'.format(mask_id)) / 255
        img_nan = np.argwhere(np.isnan(img_m))
        mask_nan = np.argwhere(np.isnan(mask))
        print(img_id + ' read')
        print("Nan values for {}: {}".format(img_id, img_nan))        
#        print("Nan values for {}: {}".format(mask, mask_nan))
        if len(np.isnan(mask)) > 0:
            print("Nan values found in {}".format(mask_id))