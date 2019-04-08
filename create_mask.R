remove(list=ls()) 
setwd("~/GitHub/keras_playground")
install.packages("raster")
install.packages("rgdal")
library(raster, rgdal)

to_mask = raster("../make_mask/pilot_area_master.tif")

matrix1 = c(-Inf, 0.8, 0, 0.9, 1.1, 1, 1.2, Inf, 0)
reclass1 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class1 = reclassify(to_mask, reclass1)
writeRaster(mask_class1, 'output/mask_class1.tif')

matrix2 = c(-Inf, 1.8, 0, 1.9, 2.1, 1, 2.2, Inf, 0)
reclass2 = matrix(matrix2, ncol = 3, byrow = TRUE)
mask_class2 = reclassify(to_mask, reclass2)
writeRaster(mask_class2, 'output/mask_class2.tif')

matrix3 = c(-Inf, 2.8, 0, 2.9, 3.1, 1, 3.2, Inf, 0)
reclass3 = matrix(matrix3, ncol = 3, byrow = TRUE)
mask_class3 = reclassify(to_mask, reclass3)
writeRaster(mask_class3, 'output/mask_class3.tif')

matrix4 = c(-Inf, 3.8, 0, 3.9, 4.1, 1, 4.2, Inf, 0)
reclass4 = matrix(matrix4, ncol = 3, byrow = TRUE)
mask_class4 = reclassify(to_mask, reclass4)
writeRaster(mask_class4, 'output/mask_class4.tif')

matrix5 = c(-Inf, 4.8, 0, 4.9, 5.1, 1, 5.2, Inf, 0)
reclass5 = matrix(matrix5, ncol = 3, byrow = TRUE)
mask_class5 = reclassify(to_mask, reclass5)
writeRaster(mask_class5, 'output/mask_class5.tif')

matrix6 = c(-Inf, 5.8, 0, 5.9, 6.1, 1, 6.2, Inf, 0)
reclass6 = matrix(matrix6, ncol = 3, byrow = TRUE)
mask_class6 = reclassify(to_mask, reclass6)
writeRaster(mask_class6, 'output/mask_class6.tif')

mask_stack = stack(mask_class1, mask_class2, mask_class3, mask_class4, mask_class5, mask_class6)
rm(mask_class1, mask_class2, mask_class3, mask_class4, mask_class5, mask_class6)
writeRaster(mask_stack, 'output/maskclass_all.tif')

# plot(mask_class1)
# plot(mask_class2)
# plot(mask_class3)
# plot(mask_class4)
# plot(mask_class5)
# plot(mask_class6)
