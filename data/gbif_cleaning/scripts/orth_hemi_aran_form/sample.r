# R script to prepare a download sample of URLs for images of orthoptera,
# hemiptera, araneae, formicidae.

# Our goal was to secure 2,000 images for each of the four categories.
# Considering that ants have approximately 212 species and Hemiptera about
# 2,000, this method provided some level of balance between the classes while
# sampling across the complete range of species-level morphological diversity.
# In the case of ants, this necessitated the random sampling of multiple images
# per species.

library(data.table)

# Read data ---------------------------------------------------------------

# The cleaned GBIF metadata table created with the script clean-gbif-urls.r
dt_clean_url <- readRDS("./data/gbif_cleaning/data_processed/dt_orth_hemi_aran_form_eu_clean_url.rds")

dt_clean_url[, sp := paste(genus, specificEpithet, sep = "_")]

# Are there any families that can have the same name across several orders?
any(duplicated(dt_clean_url[, unique(family)])) # expect FALSE

# Taxa stats --------------------------------------------------------------

# ~ Permissible licenses only ----------------------------------------------

dt_lic_ok <- copy(dt_clean_url[license_ok == TRUE])
dt_lic_ok[, uniqueN(family), by = order] # N families per order (pay attention to Hymenoptera)
# 1:     Araneae 55
# 2:  Orthoptera 13
# 3:   Hemiptera 85
# 4: Hymenoptera  1
dt_lic_ok[, uniqueN(sp), by = order] # N species per order
# 1:     Araneae  991
# 2:  Orthoptera  668
# 3:   Hemiptera 1992
# 4: Hymenoptera  212


# Sample 2021-10-28 ------------------------------------

# Our goal was to secure 2,000 images for each of the four categories.
# Considering that ants have approximately 212 species and Hemiptera about
# 2,000, this method provided some level of balance between the classes while
# sampling across the complete range of species-level morphological diversity.
# In the case of ants, this necessitated the random sampling of multiple images
# per species.

# Estimate how many images per species and how many images in total:
n_img_estim_dt <- dt_lic_ok[, uniqueN(sp), by = order][, n_img_per_sp := round(2000/V1)][, n_img := n_img_per_sp * V1]
n_img_estim_dt[]

spl_lst <- vector(mode = "list", length = nrow(n_img_estim_dt))
for (i in 1:nrow(n_img_estim_dt)){
  ord <- n_img_estim_dt[i, order]
  n_img_per_sp <- n_img_estim_dt[i, n_img_per_sp]
  spl_lst[[i]] <- dt_lic_ok[order == ord][, .SD[sample(.N, size = min(.N, n_img_per_sp), replace = FALSE)], keyby = sp]
}

spl_dt <- rbindlist(spl_lst)

# Check if we sampled what we needed. We can get smaller values because
# sometimes there were not enough available images per species.
spl_dt[, .N, by = order] 
#          order    N
# 1:     Araneae 1856
# 2:  Orthoptera 1792
# 3:   Hemiptera 1992
# 4: Hymenoptera 1474

# This should both give the same values as above if URLs are unique
spl_dt[, uniqueN(identifier), by = order] 
#          order   V1
# 1:     Araneae 1856
# 2:  Orthoptera 1792
# 3:   Hemiptera 1992
# 4: Hymenoptera 1474

spl_dt[, uniqueN(identifier)] == nrow(spl_dt) 
# must be TRUE, otherwise you risk to have same URL for multiple rows.

saveRDS(spl_dt, file = "./data/gbif_cleaning/data_processed/orth_hemi_aran_form_sample_2021_10_28.rds")
