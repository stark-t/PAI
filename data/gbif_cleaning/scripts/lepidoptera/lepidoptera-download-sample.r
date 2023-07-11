# Script to download sample of images for Lepidoptera.

library(data.table)
library(stringr)

# Sample on 2021-10-21 ----------------------------------------------

# Sample of URLs created with the script hymenoptera-sample.r
dt_spl <- readRDS(file = "./data/gbif_cleaning/data_processed/lepidoptera_sample_2021_10_21.rds")

dt_spl[, .N, by = family]
test <- dt_spl[, .N, keyby = .(family, sp)]

dt_spl[, ext := str_extract(string = identifier, pattern = "\\.jpg|\\.jpeg|\\.JPG|\\.png")]
dt_spl[, .N, by = ext]
#      ext    N
# 1:  .JPG   10
# 2: .jpeg  715
# 3:  .jpg 4365
# 4:  <NA>    1
# 5:  .png   12
dt_spl[ext == ".JPG", ext := ".jpg"]

# Check this NA. Seems to be a jpeg
dt_spl[is.na(ext)]
dt_spl[is.na(ext),  ext := ".jpeg"]

# Create image names with taxa info and IDs
dt_spl[, save_name_no_ext := paste(order, family, sp, gbifID, media_id, sep = "_")]
dt_spl[, save_name := paste(save_name_no_ext, ext, sep = "")]
dt_spl[, save_name_no_ext := NULL]

# Copy existing files. Note that you can still download all the URLs without
# using this filter. In our case, we had some images already downloaded and we
# didn't want to access the URL's again, so we just copied the existing images
# to the final destination.
nrow(dt_spl[downloaded_obs == TRUE]) # 3518 nr of img to copy
copy_path <- "./data/gbif_cleaning/gbif/samples/img_lepidoptera_sample_2021_10_21"
dt_spl[downloaded_obs == TRUE, file.copy(from = jpg_path, to = copy_path)]

# Rename these files using the gbifID and media_id instead of their URL number
# as I downloaded them first time.
old_paths <- dt_spl[, tstrsplit(jpg_path, split = "/")]
# last column contains the old file names
dt_spl[, old_name := old_paths[[ncol(old_paths)]]]
old_names <- dt_spl[downloaded_obs == TRUE, old_name]
new_names <- dt_spl[downloaded_obs == TRUE, save_name]
length(old_names) == length(new_names) # Expect TRUE
# Check some values
old_names[1]
new_names[1]
old_names[length(old_names)]
new_names[length(new_names)]
file.rename(from = file.path(copy_path, old_names),
            to = file.path(copy_path, new_names))
# Can check manually some values to see if renaming was done correctly. Compare
# visually the two files.
dt_spl[save_name == "Lepidoptera_Zygaenidae_Zygaena_viciae_2835111249_2445953.jpg"]

# Download the rest
nrow(dt_spl[downloaded_obs == FALSE]) # 1585
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
# 147.057  121.109 2261.826 ~ 38 min

err
# 150

save(err, file = "./data/gbif_cleaning/gbif/samples/img_lepidoptera_sample_2021_10_21_errors.rda")
