remove(list=ls()) 
setwd("~/GitHub/keras_playground")
install.packages("raster")
install.packages("rgdal")
library(raster, rgdal)

big_mask = raster("output/maskclass_all.tif")
big_mask_18n = projectRaster(big_mask, crs = "+proj=utm +zone=18 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

img_dir = "data/planet_training/img"
images = dir()
mask_dir = "data/planet_training/mask"

filenames = list.files(img_dir, pattern = "*.tif")
for (f in filenames){
  f_base = tools::file_path_sans_ext(f)
  r = raster(file.path(img_dir, f))
  crop*=()
  
}