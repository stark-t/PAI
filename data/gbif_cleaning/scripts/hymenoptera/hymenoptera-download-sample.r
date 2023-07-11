# Script to download sample of images for Hymenoptera (no ants).

library(data.table)
library(stringr)

# First sample on 2021-09-22 ----------------------------------------------

# Sample of URLs created with the script hymenoptera-sample.r
dt_spl <- readRDS(file = "./data/gbif_cleaning/data_processed/hymenoptera_sample_2021_09_22.rds")

dt_spl[, .N, by = family]
test <- dt_spl[, .N, keyby = .(family, sp)]

dt_spl[, ext := str_extract(string = identifier, pattern = "\\.jpg|\\.jpeg|\\.JPG|\\.png")]
dt_spl[, .N, by = ext]
#      ext    N
# 1:  .jpg 2543
# 2:  .JPG    4
# 3: .jpeg  535
# 4:  .png    8

dt_spl[ext == ".JPG", ext := ".jpg"]

# Create image names with taxa info and IDs
dt_spl[, save_name_no_ext := paste(order, family, sp, gbifID, media_id, sep = "_")]
dt_spl[, save_name := paste(save_name_no_ext, ext, sep = "")]
dt_spl[, save_name_no_ext := NULL]

# Copy existing files. Note that you can still download all the URLs without
# using this filter. In our case, we had some images already downloaded and we
# didn't want to access the URL's again, so we just copied the existing images
# to the final destination.
nrow(dt_spl[downloaded_obs == TRUE]) # nr of img to copy
# Create this folder manually
copy_path <- "./data/gbif_cleaning/gbif/samples/img_hymenoptera_sample_2021_09_22"
dt_spl[downloaded_obs == TRUE, file.copy(from = jpg_path, to = copy_path)]

# Download the rest
nrow(dt_spl[downloaded_obs == FALSE]) # 1730
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
# 42.057   42.222 7655.108 ~ 130 min
err
# [1]   43   51   56   69   71  117  132  144  198  201  207  227  233  247  262  269  276  282  330  349  354  361  381  386  388  413  420
# [28]  438  446  454  479  489  508  519  554  582  584  588  600  603  605  608  610  614  622  635  644  659  663  668  670  673  675  680
# [55]  685  709  741  750  758  781  789  797  801  810  825  841  850  858  866  878  882  885  898  900  912  947  955  968  983 1003 1025
# [82] 1033 1045 1052 1084 1095 1113 1128 1136 1141 1165 1189 1205 1210 1217 1233
save(err, file = "./data/gbif_cleaning/gbif/samples/img_hymenoptera_sample_2021_09_22_errors.rda")
