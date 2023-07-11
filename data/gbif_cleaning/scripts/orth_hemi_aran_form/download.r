# Script to download sample of images for orthoptera, hemiptera, araneae,
# formicidae.

library(data.table)
library(stringr)

# Sample on 2021_10_28 --------------------------------------------------------------

# Sample of URLs created with the script sample.r
dt_spl <- readRDS(file = "./data/gbif_cleaning/data_processed/orth_hemi_aran_form_sample_2021_10_28.rds")

dt_spl[, unique(order)]
# "Araneae"     "Orthoptera"  "Hemiptera"   "Hymenoptera"
dt_spl[, .N, by = order] 
#          order    N
# 1:     Araneae 1856
# 2:  Orthoptera 1792
# 3:   Hemiptera 1992
# 4: Hymenoptera 1474

dt_spl[, ext := str_extract(string = identifier, pattern = "\\.jpg|\\.jpeg|\\.JPG|\\.png")]
dt_spl[, .N, by = ext]
#      ext    N
# 1:  .jpg 4069
# 2: .jpeg 2953
# 3:  .JPG   34
# 4:  .png   57
# 5:  <NA>    1

# Check this NA. Seems to be a jpeg
dt_spl[is.na(ext)]
dt_spl[is.na(ext),  ext := ".jpeg"]

dt_spl[ext == ".JPG", ext := ".jpg"]

# Create image names with taxa info and IDs
dt_spl[, save_name_no_ext := paste(order, family, sp, gbifID, media_id, sep = "_")]
dt_spl[, save_name := paste(save_name_no_ext, ext, sep = "")]
dt_spl[, save_name_no_ext := NULL]


# ~ Araneae ---------------------------------------------------------------

dt <- dt_spl[order == "Araneae"]

# create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_araneae_sample_2021_10_28"

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
# 109.115  174.074 2275.044 ~ 40 min

err # 487
save(err, file = "./data/gbif_cleaning/gbif/samples/img_araneae_sample_2021_10_28_errors.rda")
rm(err)

# ~ Hemiptera ------------------------------------------------------------

dt <- dt_spl[order == "Hemiptera"]

# create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_hemiptera_sample_2021_10_28"

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
# 113.569  183.604 2263.738 ~ 40 min
err # 1426
save(err, file = "./data/gbif_cleaning/gbif/samples/img_hemiptera_sample_2021_10_28_errors.rda")
rm(err)

# ~ Hymenoptera - formicidae ------------------------------------------------------------

dt <- dt_spl[order == "Hymenoptera"]

# create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_hymenoptera_formicidae_sample_2021_10_28"

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
# 88.507  159.368 1963.755
err # NULL
save(err, file = "./data/gbif_cleaning/gbif/samples/img_hymenoptera_formicidae_sample_2021_10_28_errors.rda")
rm(err)

# ~ Orthoptera ------------------------------------------------------------

dt <- dt_spl[order == "Orthoptera"]

# create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_orthoptera_sample_2021_10_28"

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
# 130.888  207.733 2428.788 
err # NULL
save(err, file = "./data/gbif_cleaning/gbif/samples/img_orthoptera_sample_2021_10_28_errors.rda")
rm(err)
