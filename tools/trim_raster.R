remove(list=ls()) 
# install.packages("raster")
# install.packages("rgdal")

setwd("~/GitHub/untouched_gh_original/deep-unet-for-satellite-image-segmentation-master")
library(raster, rgdal)

to_mask = stack("data/20180412_143955_0e20_1B_AnalyticMS.tif")
to_match = stack("data/mband/test.tif")

plot(to_mask)
plot(to_match)

r3 = crop(to_mask, to_match)
plot(r3)

writeRaster(r3, "data/20180412_143955_0e20_1B_AnalyticMS_crop.tif")
