#!/usr/bin/env Rscript

remove(list=ls()) 
library(tiff)
library(raster)



args = commandArgs(trailingOnly=TRUE)

if (length(args)<=1) {
  stop("Two arguments must be supplied:sat_image.tiff, source_of_mask.tiff", call.=FALSE)
}

sat_image = stack(args[1])
to_mask = stack(args[2])
