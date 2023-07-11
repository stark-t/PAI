# Script to download sample of images for Diptera.

library(data.table)
library(stringr)

# Sample on 2021-09-20 ----------------------------------------------

# Sample of URLs created with the script coleoptera-sample.r
dt_spl <- readRDS(file = "./data/gbif_cleaning/data_processed/diptera_sample_2021_09_20.rds")

dt_spl[, .N, by = family]

dt_spl[, ext := str_extract(string = identifier, pattern = "\\.jpg|\\.jpeg|\\.png")]
dt_spl[, unique(ext)]
# [1] ".jpg"  ".jpeg" ".png" 
dt_spl[, .N, by = ext]
#      ext    N
# 1:  .jpg 2318
# 2: .jpeg  488
# 3:  .png    1

# Create image names with taxa info and IDs
dt_spl[, save_name_no_ext := paste(order, family, sp, gbifID, media_id, sep = "_")]
dt_spl[, save_name := paste(save_name_no_ext, ext, sep = "")]

# Copy existing files. Note that you can still download all the URLs without
# using this filter. In our case, we had some images already downloaded and we
# didn't want to access the URL's again, so we just copied the existing images
# to the final destination.
nrow(dt_spl[downloaded_obs == TRUE]) # nr of img to copy
# Create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_diptera_sample_2021_09_20"
dt_spl[downloaded_obs == TRUE, file.copy(from = jpg_path, to = copy_path)]

# Download the rest
nrow(dt_spl[downloaded_obs == FALSE])
dt <- dt_spl[downloaded_obs == FALSE]

system.time({
  setwd(copy_path)
  
  err <- c()
  for (i in 1:nrow(dt)){
    url <- dt[i, identifier]
    save_name <- dt[i, save_name]
    tryCatch(download.file(url, save_name, mode = 'wb', quiet = TRUE),
             error = function(e){ 
               err <<- c(err, i)
             })
  }
})
# user   system  elapsed 
# 108.610  296.019 1468.223 

err
# 231 download errors

save(err, file = "./data/gbif_cleaning/gbif/samples/err_img_diptera_sample_2021_09_20.rda")
