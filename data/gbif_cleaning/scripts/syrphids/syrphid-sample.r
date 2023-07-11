# Take a sample of around 1000 Syrphidae that we will use for extra testing. We
# want to see if the model confuses them with hymenoptera.

library(bit64)
library(data.table)
library(magrittr)

dt_clean_url <- readRDS("./data/gbif_cleaning/data_processed/dt_diptera_eu_clean_url.rds")

# Make sure you have bit64 package installed so that you have the correct
# representation of gbifID.

dt_clean_url[, sp := paste(genus, specificEpithet, sep = "_")]

# This was the Diptera sample already prepared. We need to avoid sampling from these.
dt_sampled <- readRDS(file = "./data/gbif_cleaning/data_processed/diptera_sample_2021_09_20.rds")

dt_clean_url[, downloaded := identifier %in% dt_sampled$identifier]
dt_clean_url[downloaded == TRUE, .N] # 2807

dt_clean_url[, .N, keyby = family] # to check if there are any strange names. All good

# N urls by family
dt_clean_url[downloaded == FALSE][license_ok == TRUE][, .N, keyby = family]
#            family      N
# 1:   Anthomyiidae   2015
# 2:    Bombyliidae  15871 
# 3:  Calliphoridae  10519
# 4:      Conopidae   4721
# 5:      Empididae   4218
# 6:      Fanniidae    194
# 7:      Hybotidae    493
# 8:       Muscidae  13660
# 9:  Sarcophagidae    161
# 10: Scathophagidae   6545
# 11:       Sepsidae    748
# 12:  Stratiomyidae  16047
# 13:      Syrphidae 201622 # mimic Hymenoptera
# 14:      Tabanidae   5633
# 15:     Tachinidae  23644

dt_syrph <- dt_clean_url[downloaded == FALSE][license_ok == TRUE][family == "Syrphidae"]

dt_syrph$sp %>% unique() %>% length() # 383 unique species
dt_syrph$genus %>% unique() %>% length() # 72 unique genera

# Some species have thousands of URLs and other have just one URL
dt_syrph[, .N, by = sp][order(-N)]

# Having 383 sp, my first guess was to go with max 3 URLs per species when
# sampling so to arrive at about 1000 images that we can download.
n_img_per_sp <- 3
set.seed(123)
dt_spl_syrph <- dt_syrph[, .SD[sample(.N, size = min(.N, n_img_per_sp), replace = FALSE)], keyby = sp]
nrow(dt_spl_syrph) # 1093
# Keep in mind that about 15% of these images might need to be discarded
# (larvae, body parts, etc).

saveRDS(dt_spl_syrph, file = "./data/gbif_cleaning/data_processed/syrphidae_sample_2022_06_17.rds")
