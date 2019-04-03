remove(list=ls()) 
setwd("~/GitHub/keras_playground")
install.packages("raster")
install.packages("rgdal")
library(raster, rgdal)

# args = commandArgs(trailingOnly=TRUE)
# 
# if (length(args)<=1) {
#   stop("Two arguments must be supplied:sat_image.tiff, source_of_mask.tiff", call.=FALSE)
# }
# sat_image = stack(args[1])
# to_mask = stack(args[2])

sat_image = raster("data/planet_training/img/20180412_143155_1003_1B_AnalyticMS.tif")
to_mask = raster("../make_mask/pilot_area_master.tif")
