# Script to download sample of images for Coleoptera.

library(data.table)
library(stringr)

# Sample on 2022-04-06 ----------------------------------------------

# Sample of URLs created with the script coleoptera-sample.r
dt_spl <- readRDS(file = "./data/gbif_cleaning/data_processed/coleoptera_sample_2022_04_06.rds")

dt_spl[, .N, keyby = family]
test <- dt_spl[, .N, keyby = .(family, sp)]

dt_spl[, ext := str_extract(string = identifier, pattern = "\\.jpg|\\.jpeg|\\.JPG|\\.png")]
dt_spl[, .N, by = ext]
#      ext    N
# 1:  .jpg 1214
# 2: .jpeg 1244
# 3:  .png   18
# 4:  .JPG   14
# 5:  <NA>    1

dt_spl[ext == ".JPG"]
dt_spl[ext == ".JPG", ext := ".jpg"]

# Check this NA. Seems to be a jpeg
dt_spl[is.na(ext)] # Looks like being larvae, so drop the img
# dt_spl[is.na(ext),  ext := ".jpeg"]
dt_spl <- dt_spl[!is.na(ext)]

dt_spl[, .N, by = ext]
#      ext    N
# 1:  .jpg 1228
# 2: .jpeg 1244
# 3:  .png   18

# Create image names with taxa info and IDs
dt_spl[, save_name_no_ext := paste(order, family, sp, gbifID, media_id, sep = "_")]
dt_spl[, save_name := paste(save_name_no_ext, ext, sep = "")]
dt_spl[, save_name_no_ext := NULL]

# Download

# Create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_coleoptera_sample_2022_04_21"

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

err
# NULL = waw! no download errors

save(err, file = "./data/gbif_cleaning/gbif/samples/img_coleoptera_sample_2022_04_21_errors.rda")
