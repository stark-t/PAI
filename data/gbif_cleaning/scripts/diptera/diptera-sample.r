# R script to prepare a download sample of URLs for images of Diptera. For
# families consisting of more than 200 species, we applied random sampling to
# select 200 species, choosing one image per species. For families with fewer
# than 200 species, we allowed the inclusion of multiple random images per
# species.

library(data.table)
library(tidyverse)

# Read data ---------------------------------------------------------------

# The cleaned GBIF metadata table created with the script diptera-clean-gbif-urls.r
dt_clean_url <- readRDS("./data/gbif_cleaning/data_processed/dt_diptera_eu_clean_url.rds")
# Note that some were downloaded previously from observation.org 
# We only have these locally.
dt_down <- readRDS("./data/gbif_cleaning/data_processed/all_orders_downloaded.rds") 
dt_down <- dt_down[order == "Diptera"]

# Mark those downloaded already -------------------------------------------

dt_clean_url[, downloaded_obs := identifier %in% dt_down$identifier]
dt_clean_url[downloaded_obs == TRUE, .N] # 96651
dt_clean_url[, sp := paste(genus, specificEpithet, sep = "_")]

# Check changes in species synonyms/names since the last download from
# observation.org. These records have the same URL (identifier), but taxa info
# has changed (is different). So, use the most recent taxa info. For the files
# that were already downloaded, that means you have to rename the files! Sorry,
# I discovered that names updated in these databases at a later stage.
dt_test <- merge(dt_clean_url, dt_down, by = "identifier")
dt_test[, sp.y := paste(genus.y, specificEpithet.y, sep = "_")]
dt_sp_dif <- dt_test[sp != sp.y, .(identifier, sp, sp.y, family.x, family.y)]
sp_dif <- unique(dt_sp_dif, by = c("sp", "sp.y")) # ignore identifier
sp_dif[, identifier := NULL] # delete it so that it doesn't create confusion
dt_sp_dif[, .N, by = .(sp, sp.y)] # if you need the number of when such things occur
# If you check at https://fauna-eu.org/ you can find that the older names (sp.y)
# from observation.org are outdated or synonyms.
setnames(sp_dif, new = c("sp_new", "sp_old", "family_new", "family_old"))
write.csv(sp_dif, "./data/gbif_cleaning/data_processed/diptera_taxa_changes.csv", row.names = FALSE)


# Taxa stats --------------------------------------------------------------
  
# ~ All license data -----------------------------------------------------

dt_all_lic <- copy(dt_clean_url)

dt_all_lic[, n_sp_in_family := uniqueN(sp), by = family]
dt_all_lic[, n_genera_in_family := uniqueN(genus), by = family]
dt_all_lic[, n_url_in_family := .N, by = family]
dt_all_lic[, n_img_down_in_family := sum(downloaded_obs), by = family]

dt_fam_stats <- unique(dt_all_lic, by = "family")
dt_fam_stats <- dt_fam_stats[,.(family, n_sp_in_family, n_genera_in_family,
                                n_url_in_family, n_img_down_in_family)]
dt_fam_stats[, prc_sp := n_sp_in_family / sum(n_sp_in_family) * 100]
dt_fam_stats[, prc_gen := n_genera_in_family / sum(n_genera_in_family) * 100]
dt_fam_stats[, prc_url := n_url_in_family / sum(n_url_in_family) * 100]
dt_fam_stats[, license := "all"]
dt_fam_stats <- dt_fam_stats %>% arrange(-prc_url)
dt_fam_stats

dt_fam_stats %>% select(family:n_genera_in_family) %>% pivot_longer(cols = !family) %>% 
  ggplot(aes(y=value, x=family, fill = name)) + 
  geom_bar(stat = "identity") +
  labs(x = NULL) +
  scale_x_discrete(limits= dt_fam_stats %>% arrange(-n_sp_in_family) %>% pull(family)) +
  theme(axis.text.x=element_text(angle = 90, vjust = 0.5))

dt_fam_stats %>% select(family, prc_sp:prc_url) %>% pivot_longer(cols = !family) %>% 
  ggplot(aes(y=value, x=family, fill = name)) + 
  geom_bar(stat = "identity") +
  labs(x = NULL) +
  scale_x_discrete(limits = dt_fam_stats %>% arrange(-prc_url) %>% pull(family)) +
  theme(axis.text.x=element_text(angle = 90, vjust = 0.5))


# ~ Permissible licenses only ----------------------------------------------

dt_lic_ok <- copy(dt_clean_url[license_ok == TRUE])

dt_lic_ok[, n_sp_in_family := uniqueN(sp), by = family]
dt_lic_ok[, n_genera_in_family := uniqueN(genus), by = family]
dt_lic_ok[, n_url_in_family := .N, by = family]
dt_lic_ok[, n_img_down_in_family := sum(downloaded_obs), by = family]

dt_fam_stats_lic_ok <- unique(dt_lic_ok, by = "family")
dt_fam_stats_lic_ok <- dt_fam_stats_lic_ok[,.(family, n_sp_in_family, n_genera_in_family,
                                              n_url_in_family, n_img_down_in_family)]
dt_fam_stats_lic_ok[, prc_sp := n_sp_in_family / sum(n_sp_in_family) * 100]
dt_fam_stats_lic_ok[, prc_gen := n_genera_in_family / sum(n_genera_in_family) * 100]
dt_fam_stats_lic_ok[, prc_url := n_url_in_family / sum(n_url_in_family) * 100]
dt_fam_stats_lic_ok[, license := "permisible"]
dt_fam_stats_lic_ok <- dt_fam_stats_lic_ok %>% arrange(-prc_url)
dt_fam_stats_lic_ok
dt_fam_stats_lic_ok[, .(family, n_url_in_family, n_sp_in_family)]

dt_fam_stats_lic_ok %>% select(family:n_genera_in_family) %>% pivot_longer(cols = !family) %>% 
  ggplot(aes(y=value, x=family, fill = name)) + 
  geom_bar(stat = "identity") +
  labs(x = NULL) +
  scale_x_discrete(limits= dt_fam_stats_lic_ok %>% arrange(-n_sp_in_family) %>% pull(family)) +
  theme(axis.text.x=element_text(angle = 90, vjust = 0.5))

dt_fam_stats_lic_ok %>% select(family, prc_sp:prc_url) %>% pivot_longer(cols = !family) %>% 
  ggplot(aes(y=value, x=family, fill = name)) + 
  geom_bar(stat = "identity") +
  labs(x = NULL) +
  scale_x_discrete(limits = dt_fam_stats_lic_ok %>% arrange(-prc_url) %>% pull(family)) +
  theme(axis.text.x=element_text(angle = 90, vjust = 0.5))


# ~ Differences due to license --------------------------------------------

dt_fam_stats_rbind <- rbindlist(list(dt_fam_stats, dt_fam_stats_lic_ok))
# How many species we lose due to license?
dt_dif <- dt_fam_stats_rbind[, .( n_sp_dif = abs(diff(n_sp_in_family)),
                                  prc_sp_dif = abs(diff(n_sp_in_family)) / sum(n_sp_in_family) * 100,
                                  n_gen_dif = abs(diff(n_genera_in_family)),
                                  n_url_dif = abs(diff(n_url_in_family)),
                                  prc_ulr_dif = abs(diff(n_url_in_family)) / sum(n_url_in_family) * 100,
                                  n_img_down_dif = abs(diff(n_img_down_in_family)),
                                  prc_img_down_dif = abs(diff(n_img_down_in_family)) / sum(n_img_down_in_family) * 100 ), 
                             by = family]
dt_dif
cbind(dt_fam_stats, dt_dif)

# Sample 2021-09-20 ------------------------------------------------------------------

# For families consisting of more than 200 species, we applied random sampling
# to select 200 species, choosing one image per species. For families with fewer
# than 200 species, we allowed the inclusion of multiple random images per
# species.

# When taking sample you need to account for those downloaded already

f_spl_lst <- vector(mode = "list", length = nrow(dt_fam_stats_lic_ok))
spl_thrs <- 200

# First check if you can achieve your sample from those downloaded and if you
# need to download further.

for (i in 1:nrow(dt_fam_stats_lic_ok)){
  f <- dt_fam_stats_lic_ok$family[i]
  f_sp <- dt_lic_ok[family == f, unique(sp)]
  n_sp <- length(f_sp)
  f_sp_down <- dt_lic_ok[downloaded_obs == TRUE][family == f, unique(sp)]
  n_sp_down <- length(f_sp_down)
  # if character(0), be careful
  
  # if n_sp_down >= spl_thrs then
  #   take sample from downloaded species, 1 image per species
  # else if n_sp_down < spl_thrs 
  #   take sample from downloaded and complete sample with new species urls
  
  seed <- 2021-9-20 # change date when you sample again
  set.seed(seed)
  sp_spl <- sample(f_sp, size = min(n_sp, spl_thrs))
  sp_down_spl <- sample(f_sp_down, size = min(n_sp_down, spl_thrs))
  
  if (n_sp_down >= spl_thrs) {
    # put the family filter, just in case two species of the same name can be
    # from different families.
    set.seed(seed)
    # Sample 1 img per species
    f_spl <- dt_lic_ok[downloaded_obs == TRUE][family == f][sp %in% sp_down_spl][, .SD[sample(.N, size = 1, replace = FALSE)], keyby = sp]
    f_spl <- f_spl[,.(media_id, gbifID, order, family, sp, identifier, downloaded_obs)]
  } else {
    # Distribute the sample size by number of species
    f_sp_union <- union(f_sp, f_sp_down)
    new_sp <- setdiff(f_sp, f_sp_down)
    n_img_spl <- ceiling(spl_thrs/length(f_sp_union))
    set.seed(seed)
    f_spl_down <- dt_lic_ok[downloaded_obs == TRUE][family == f][sp %in% sp_down_spl][, .SD[sample(.N, size = min(.N, n_img_spl), replace = FALSE)], keyby = sp]
    # f_spl_down[, .N, by = sp]
    set.seed(seed)
    f_spl_new_url <- dt_lic_ok[downloaded_obs == FALSE][family == f][sp %in% new_sp][, .SD[sample(.N, size = min(.N, n_img_spl), replace = FALSE)], keyby = sp]
    # f_spl_new_url[, .N, by = sp]
    f_spl <- rbindlist(list(f_spl_down, f_spl_new_url))
    f_spl <- f_spl[,.(media_id, gbifID, order, family, sp, identifier, downloaded_obs)]
  }
  f_spl_lst[[i]] <- f_spl
}

f_spl_dt <- rbindlist(f_spl_lst)
f_spl_dt <- merge(f_spl_dt, dt_down[, .(identifier, jpg_path)], by = "identifier", all.x = TRUE)
saveRDS(f_spl_dt, file = "./data/gbif_cleaning/data_processed/diptera_sample_2021_09_20.rds")
