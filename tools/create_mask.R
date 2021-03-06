remove(list=ls()) 
install.packages("raster")
install.packages("rgdal")

setwd("~/GitHub/keras_playground")
library(raster, rgdal)

to_mask = raster("../make_mask/pilot_area_master.tif")

matrix1 = c(NA, NA, 0, -Inf, 0.8, 0, 0.9, 1.1, 1, 1.2, Inf, 0)
matrix2 = c(NA, NA, 0, -Inf, 1.8, 0, 1.9, 2.1, 1, 2.2, Inf, 0)
matrix3 = c(NA, NA, 0, -Inf, 2.8, 0, 2.9, 3.1, 1, 3.2, Inf, 0)
matrix4 = c(NA, NA, 0, -Inf, 3.8, 0, 3.9, 4.1, 1, 4.2, Inf, 0)
matrix5 = c(NA, NA, 0, -Inf, 4.8, 0, 4.9, 5.1, 1, 5.2, Inf, 0)
matrix6 = c(NA, NA, 0, -Inf, 5.8, 0, 5.9, 6.1, 1, 6.2, Inf, 0)

reclass1 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class1 = reclassify(to_mask, reclass1)

reclass2 = matrix(matrix2, ncol = 3, byrow = TRUE)
mask_class2 = reclassify(to_mask, reclass2)

reclass3 = matrix(matrix3, ncol = 3, byrow = TRUE)
mask_class3 = reclassify(to_mask, reclass3)

reclass4 = matrix(matrix4, ncol = 3, byrow = TRUE)
mask_class4 = reclassify(to_mask, reclass4)

reclass5 = matrix(matrix5, ncol = 3, byrow = TRUE)
mask_class5 = reclassify(to_mask, reclass5)

reclass6 = matrix(matrix6, ncol = 3, byrow = TRUE)
mask_class6 = reclassify(to_mask, reclass6)

#search for infinte values in the rasters
# which(is.infinite(mask_class1@data@values))
# which(is.infinite(mask_class2@data@values))
# which(is.infinite(mask_class3@data@values))
# which(is.infinite(mask_class4@data@values))
# which(is.infinite(mask_class5@data@values))
# which(is.infinite(mask_class6@data@values))
#search for NA values in the rasters
# which(is.na(mask_class1@data@values))
# which(is.na(mask_class2@data@values))
# which(is.na(mask_class3@data@values))
# which(is.na(mask_class4@data@values))
# which(is.na(mask_class5@data@values))
# which(is.na(mask_class6@data@values))

writeRaster(mask_class1, 'output/mask_class1.tif', datatype='INT4S')
writeRaster(mask_class2, 'output/mask_class2.tif', datatype='INT4S')
writeRaster(mask_class3, 'output/mask_class3.tif', datatype='INT4S')
writeRaster(mask_class4, 'output/mask_class4.tif', datatype='INT4S')
writeRaster(mask_class5, 'output/mask_class5.tif', datatype='INT4S')
writeRaster(mask_class6, 'output/mask_class6.tif', datatype='INT4S')

mask_stack = stack(mask_class1, mask_class2, mask_class3, mask_class4, mask_class5, mask_class6)
rm(to_mask, mask_class1, mask_class2, mask_class3, mask_class4, mask_class5, mask_class6)
writeRaster(mask_stack, 'output/maskclass_all.tif', datatype='INT4S')

#create some diagnostic plots
# plot(mask_class1)
# plot(mask_class2)
# plot(mask_class3)
# plot(mask_class4)
# plot(mask_class5)
# plot(mask_class6)
