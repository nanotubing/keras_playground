import math, os.path, sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

from train_unet import weights_path, get_model, normalize, PATCH_SZ, N_CLASSES

def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        #only print out debug info if flag is set
        if x0_x1_debug == True:
            print("K: {}".format(k))
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        #only print out debug info if flag is set
        if x0_x1_debug == True:
            print("x0: {}, x1: {}".format(x0, x1))
        #only print out debug info if flag is set
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        #only print out debug info if flag is set
        if x0_x1_debug == True:
            print("y0: {}, y1: {}".format(y0, y1))
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict


if __name__ == '__main__':
    #are we running the predictions script against the built-in test data, or
    #a planet image?
#    planet_test = False
    planet_test = True
    #set debug flag for additional output to help fix predict function
    x0_x1_debug = False
    model = get_model()
    model.load_weights(weights_path)
    
    overwrite_check = ['output/planet_classtest.tif', 'output/result.tif', 'output/map.tif', 'output/planet_result.tif', 'output/planet_map.tif']
    for file in overwrite_check:
        if os.path.exists(file):
            print('ERROR: file {0} already exists. Please rename or delete the following files before creating predictions:'.format(file))
            print(*overwrite_check, sep = '\n')
            sys.exit()
            
    if planet_test == False:
        image_id = 'test'
        weights_path = 'weights/150_epoch_unet_weights.hdf5'
        img = normalize(tiff.imread('data/mband/{}.tif'.format(image_id)).transpose([1,2,0]))   # make channels last
    elif planet_test == True:
        planet_imagedir = 'data/planet_training/predict/'
        image_id = '20180412_143154_1003_1B_AnalyticMS'
        # rearranging order for planet image no longer necessary now that it matches training data
        img = normalize(tiff.imread(planet_imagedir+'{}.tif'.format(image_id)))
        #    add 4 channels of 0 to array to predict planet image
#        img_pad = ((0,0), (0,0), (0,4))
#        img_fixed2 = np.pad(img, pad_width=img_pad, mode='constant', constant_values=0)
        #trim the planet image to the same dimensions as training data
#        img_fixed2 = img_fixed2[:848, :837, :]
#        tiff.imsave('output/planet_classtest.tif', img)
#        img = img_fixed2
        
    for i in range(7):
        if i == 0:  # reverse first dimension
            mymat = predict(img[::-1,:,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print("Case 1",img.shape, mymat.shape)
        elif i == 1:    # reverse second dimension
            temp = predict(img[:,::-1,:], model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print("Case 2", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp[:,::-1,:], mymat ]), axis=0 )
        elif i == 2:    # transpose(interchange) first and second dimensions
            #transpose removed to hopefully unbreak script 4/16/19
            temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print("Case 3", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp, mymat ]), axis=0 )
        elif i == 3:
            #was previously rotating by 90 deg. This and 180 deg rotation does not work with 
            #rectangular images like ours.
            #circle back and add augmentation after run completes
            temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print("Case 4", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp, mymat ]), axis=0 )
        elif i == 4:
            temp = predict(np.rot90(img,2), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print("Case 5", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ np.rot90(temp,-2).transpose([2,0,1]), mymat ]), axis=0 )
        elif i == 5:
            temp = predict(np.rot90(img,3), model, patch_sz=PATCH_SZ, n_classes=N_CLASSES)
            print("Case 6", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ np.rot90(temp, -3).transpose(2,0,1), mymat ]), axis=0 )
        else:
            temp = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
            print("Case 7", temp.shape, mymat.shape)
            mymat = np.mean( np.array([ temp, mymat ]), axis=0 )
     
        #create classified map
#    map = picture_from_mask(mymat, 0.5)
    
    if planet_test == False:
        tiff.imsave('output/result.tif', (255*mymat).astype('uint8'))
        tiff.imsave('output/map.tif', map)
    elif planet_test == True:
        tiff.imsave('output/planet_result.tif', (255*mymat).astype('uint8'))
        tiff.imsave('output/planet_map.tif', map)
    
