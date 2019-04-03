remove(list=ls()) 
setwd("~/GitHub/keras_playground")
install.packages("raster")
install.packages("rgdal")
library(raster, rgdal)

sat_image = raster("data/planet_training/img/20180412_143155_1003_1B_AnalyticMS.tif")
to_mask = raster("../make_mask/pilot_area_master.tif")

matrix1 = c(-Inf, 0.8, 0, 0.9, 1.1, 1, 1.2, Inf, 0)
reclass1 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class1 = reclassify(to_mask, reclass1)

matrix2 = c(-Inf, 1.8, 0, 1.9, 2.1, 1, 2.2, Inf, 0)
reclass2 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class2 = reclassify(to_mask)

matrix3 = c(-Inf, 2.8, 0, 2.9, 3.1, 1, 3.2, Inf, 0)
reclass3 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class3 = reclassify(to_mask)

matrix4 = c(-Inf, 3.8, 0, 3.9, 4.1, 1, 4.2, Inf, 0)
reclass4 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class4 = reclassify(to_mask)

matrix5 = c(-Inf, 4.8, 0, 4.9, 5.1, 1, 5.2, Inf, 0)
reclass5 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class5 = reclassify(to_mask)

matrix5 = c(-Inf, 4.8, 0, 4.9, 5.1, 1, 5.2, Inf, 0)
reclass5 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class5 = reclassify(to_mask)

matrix6 = c(-Inf, 5.8, 0, 5.9, 6.1, 1, 6.2, Inf, 0)
reclass6 = matrix(matrix1, ncol = 3, byrow = TRUE)
mask_class6 = reclassify(to_mask)
