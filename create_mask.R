remove(list=ls()) 
setwd("~/GitHub/keras_playground")
install.packages("raster")
install.packages("rgdal")
library(raster, rgdal)

sat_image = raster("data/planet_training/img/20180412_143155_1003_1B_AnalyticMS.tif")
to_mask = raster("../make_mask/pilot_area_master.tif")
