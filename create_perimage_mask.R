remove(list=ls()) 
setwd("~/GitHub/keras_playground")
install.packages("raster")
install.packages("rgdal")
library(raster, rgdal)

big_mask = raster("output/maskclass_all.tif")
img_dir = "data/planet_training/img"
mask_dir = "data/planet_training/mask/"
planet_crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
#planet_crs = '+init=EPSG:4326'

filenames = list.files(img_dir, pattern = "*.tif")
bm_target = raster(file.path(img_dir, filenames[1]), crs = planet_crs)
big_mask_proj = projectRaster(big_mask, bm_target)

for (f in filenames){
  f_base = tools::file_path_sans_ext(f)
  r = raster(file.path(img_dir, f), crs = planet_crs)
  r_proj = projectRaster(r, big_mask)
  #r_proj_ext = extent(r_proj)
  mask_cropped = crop(big_mask_proj, r)
  raster_name = paste(mask_dir, f_base, "_mask", ".tif", sep = '')
  writeRaster(mask_cropped, paste(mask_dir, f_base, "_mask", ".tif", sep = '', overwrite = TRUE))
}