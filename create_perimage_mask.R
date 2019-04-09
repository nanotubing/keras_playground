remove(list=ls()) 
install.packages("raster")
install.packages("rgdal")
setwd("~/GitHub/keras_playground")
library(raster, rgdal)

big_mask = raster("output/maskclass_all.tif")
img_dir = "data/planet_training/img"
mask_dir = "data/planet_training/mask/"
#planet_crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
planet_crs = '+init=EPSG:4326'
filenames = list.files(img_dir, pattern = "*.tif")

# bm_target = raster(file.path(img_dir, filenames[1]), crs = planet_crs)
# big_mask_proj = projectRaster(big_mask, bm_target)
#f =  "20180104_143241_0f52_1B_AnalyticMS.tif"

for (f in filenames){
  f_base = tools::file_path_sans_ext(f)
  r = raster(file.path(img_dir, f), crs = planet_crs)
  r_proj = projectRaster(r, big_mask)
  mask_cropped = crop(big_mask, r_proj)
  raster_name = paste(mask_dir, f_base, "_mask", ".tif", sep = '')
  writeRaster(mask_cropped, paste(mask_dir, f_base, "_mask", ".tif", sep = '', overwrite = TRUE))
}
