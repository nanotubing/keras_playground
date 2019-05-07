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

# r4 = unstack(r3)

list2env(setNames(unstack(r3), names(r3)), .GlobalEnv)

r4 <- stack()

r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.1)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.2)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.3)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.4)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.1)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.2)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.3)
r4 <- stack(r4, X20180412_143955_0e20_1B_AnalyticMS.4)

writeRaster(r4, "data/20180412_143955_0e20_1B_AnalyticMS_crop.tif")
