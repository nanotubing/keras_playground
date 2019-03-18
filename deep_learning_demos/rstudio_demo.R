#install SW
#recommended GH install didnt work bc proxy
#manually installed .zip from cran
install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)
#install_keras(method = "conda")
install_keras()

