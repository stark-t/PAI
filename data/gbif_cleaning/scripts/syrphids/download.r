# Script to download sample of images for the extra syrphidae test dataset.

library(bit64)
library(data.table)
library(stringr)

dt_spl <- readRDS(file = "./data/gbif_cleaning/data_processed/syrphidae_sample_2022_06_17.rds")

dt_spl[, ext := str_extract(string = identifier, pattern = "\\.jpg|\\.jpeg|\\.JPG|\\.png")]
dt_spl[, .N, by = ext]
#      ext   N
# 1:  .jpg 825
# 2: .jpeg 262
# 3:  .JPG   3
# 4:  .png   3

# Create image names with taxa info and IDs
dt_spl[, save_name_no_ext := paste(order, family, sp, gbifID, media_id, sep = "_")]
dt_spl[, save_name := paste(save_name_no_ext, ext, sep = "")]
dt_spl[, save_name_no_ext := NULL]

# Download

# Create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_syrphidae_sample_2022_06_17"

system.time({
  setwd(copy_path)
  
  err <- c()
  for (i in 1:nrow(dt_spl)){
    url <- dt_spl[i, identifier]
    save_name <- dt_spl[i, save_name]
    tryCatch(download.file(url, save_name, mode = 'wb', quiet = TRUE),
             error = function(e){ 
               err <<- c(err, i)
             })
  }
})
#    user   system  elapsed 
# 178.493  228.415 2285.780  ~ 40 min

err # [1] 349 983
save(err, file = "./data/gbif_cleaning/gbif/samples/img_syrphidae_sample_2022_06_17.rda")
rm(err)