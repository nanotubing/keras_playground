# Keras and Tensorflow for creating a classified map
We are developing a pipeline using Python, Keras, and Tensorflow to classify satellite images from [Planet](https://www.planet.com). This work was performed with the [Remote Sensing and Sustainability Lab](http://rsensus.org/en/). This served as the Capstone Project of my [Professional Science Master's in GIS](https://bulletin.temple.edu/graduate/scd/cla/geographic-information-systems-psm/) at Temple University.

The full report of this work will eventually be documented on [my portfolio](https://claudeschrader.com) and at [rsensus.org](http://rsensus.org/en/).

* train_unet.py is used to build the model. There are a number of configurable parameters such as number of Bands, number of classes, image size, patch size, and number of epochs.

* predict.py is used to create predictions from an existing model, in this case classifying satellite images. It has a few configurable parameters, such as Image Directory, Image ID to predict against, and there is a debug flag that can enable more output to assist with troubleshooting.

* gen_patches.py and unet_model.py are called by the two scripts above. It is not necessary to call these directly.

* the tools directory contains a number of smaller utility scripts used in this research

This code is based on a fantastic [unet example](https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation) designed for performing image segmentation on satellite imagery. Thanks to @reachsumit 
