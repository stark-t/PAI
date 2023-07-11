# R script to prepare a download sample of URLs for images of Coleoptera. In the
# case of Coleoptera, we had a list of genera that we considered as flower
# visitors and pollinators. There were a total 1300 species. In this case, since
# the pool of species was smaller than in the case of other orders, we decided
# to sample at least 2 images per species in order to sample the entire species
# morphological diversity. Coleoptera is known to be a very diverse order.

library(data.table)
library(magrittr)

# Read data ---------------------------------------------------------------

# This GBIF taxa backbone was needed to double check species names. It was
# created with the script gbif_backbone.r from the raw GBIF backbone.
dt_gbif_backbone <- readRDS(file = "./data/gbif_cleaning/gbif/backbone/dt_taxon.rda")
dt_gbif_backbone[, sp := paste(genus, specificEpithet, sep = "_")]

# The cleaned GBIF metadata table created with the script coleoptera-clean-gbif-urls.r
dt_clean_url <- readRDS("./data/gbif_cleaning/data_processed/dt_coleoptera_clean_url_all_euopean_taxa.rds")
dt_clean_url[, sp := paste(genus, specificEpithet, sep = "_")]

# Read table of genera considered flower visitors and pollinators in the order Coleoptera
dt_poll <- read.csv(file = "./data/gbif_cleaning/taxonomy/coleoptera_final.csv",
                    na.strings = c("NA", ""),
                    colClasses = "character")
setDT(dt_poll)
dt_poll[, .N, by = decision_vs]
#    decision_vs   N
# 1:         yes 212
# 2:          no  24
dt_poll[decision_vs == "yes", .N, by = taxonomicStatus] # all 212 cases have accepted taxonomicStatus
dt_poll <- dt_poll[decision_vs == "yes", .(gbif_family, gbif_genus)]
dt_poll[duplicated(gbif_genus)] # 0 rows
# expect no duplicates in gbif_genus as I manually checked for them

# Are there pollinator genera that we do not have in the cleaned GBIF occurrence
# table?
dt_poll[!gbif_genus %in% unique(dt_clean_url$genus)]
#     gbif_family  gbif_genus
# 1: Cerambycidae    Anisorus
# 2: Cerambycidae    Glaphyra
# 3:   Elateridae  Limoniscus
# 4:   Elateridae Megapenthes
# 5:  Nitidulidae  Fabogethes


# Taxa stats --------------------------------------------------------------

# ~ Permissible licenses only ----------------------------------------------

dt_lic_ok <- copy(dt_clean_url[license_ok == TRUE])

# Do we lose further genera by using only permissible licenses?
dt_poll[!gbif_genus %in% unique(dt_lic_ok$genus)]
#     gbif_family  gbif_genus
# 1: Cerambycidae    Anisorus
# 2: Cerambycidae    Glaphyra
# 3:   Elateridae  Limoniscus
# 4:   Elateridae Megapenthes
# 5:   Elateridae  Procraerus
# 6:  Malachiidae    Nepachys
# 7: Melandryidae    Wanachia
# 8:  Nitidulidae  Fabogethes
# 9:  Nitidulidae    Pocadius
# 10:  Phalacridae     Stilbus

# Keep data only for the family-genus combination from dt_poll
dt_lic_ok <- merge(x = dt_lic_ok,
                   y = dt_poll,
                   by.x = c("family", "genus"),
                   by.y = c("gbif_family", "gbif_genus"))


dt_lic_ok$family %>% unique %>% length() # 34 families
# Nb. urls per fam-genera
dt_family_genus_stats <- dt_lic_ok[, .(n_url = .N, n_sp = uniqueN(sp)), keyby = .(family, genus)]
dt_family_genus_stats[duplicated(genus)] # expect 0
# There are 202 genus cases, from just 1 URL to > 100k URLs. Some genera
# have many species. Most of them have less than 10 sp. 

hist(dt_family_genus_stats$n_sp)

dt_sp_stats <- dt_lic_ok[, .(n_url = .N), keyby = .(family, genus, sp)]
nrow(dt_sp_stats) # There are 1309 unique species
# In the case of Coleoptera, we sample images per species as we did with some of
# the other orders as well. We can take about 2 images per species considering
# there are about 1309 species in the final list. This will sample also the
# morphological diversity of the pollinator genera.

# There are typos or synonyms in species names! You can use the GBIF backbone to
# check the synonyms.

# For example
# Aphodius_conjugatus rename to Aphodius_coniugatus 
# Cerambyx_scopoli rename to Cerambyx_scopolii 


# Check against the backbone ----------------------------------------------

dt_sp_stats[duplicated(sp)] # expect 0 cases

nrow(dt_sp_stats) # 1309
dt_sp_stats <- merge(x = dt_sp_stats,
                     y = dt_gbif_backbone[order == 'Coleoptera' & 
                                            taxonRank == 'species' & 
                                            taxonomicStatus == 'accepted',
                                          .(family, genus, sp, taxonomicStatus)],
                     by = c('family', 'genus', 'sp'),
                     all.x = TRUE)
nrow(dt_sp_stats) # 1309, should be unchanged

dt_sp_stats[, .N, keyby = taxonomicStatus]
#    taxonomicStatus    N
# 1:            <NA>   55
# 2:        accepted 1254

dt_sp_stats[is.na(taxonomicStatus)]

# Look at the NA ones and check for typos:
View(dt_sp_stats) # look at the NA ones, but keep all the rows so to see if they are similar with their neighbors

# Some obvious cases: In these situations, use the accepted name and combine the
# URLs of both names after a visual check on gbif gallery to be sure there are
# indeed the same species:

dt_gbif_backbone[sp == 'Aphodius_conjugatus', .(family, sp, taxonomicStatus)] # NA
dt_gbif_backbone[sp == 'Aphodius_coniugatus', .(family, sp, taxonomicStatus)] # accepted
# Both names have URLs but Aphodius_coniugatus is the accepted case and has more URLs
dt_lic_ok[sp == 'Aphodius_conjugatus'] # 3 img
dt_lic_ok[sp == 'Aphodius_coniugatus'] # 16 img
dt_sp_stats[sp == 'Aphodius_conjugatus', ':=' (sp = 'Aphodius_coniugatus', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Aphodius_conjugatus', sp := 'Aphodius_coniugatus']


dt_gbif_backbone[sp == 'Cerambyx_scopoli', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Cerambyx_scopolii', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Cerambyx_scopoli'] # just 1 img
dt_lic_ok[sp == 'Cerambyx_scopolii'] # 1562 img
dt_sp_stats[sp == 'Cerambyx_scopoli', ':=' (sp = 'Cerambyx_scopolii', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Cerambyx_scopoli', sp := 'Cerambyx_scopolii']


dt_gbif_backbone[sp == 'Chrysomela_tremulae', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Chrysomela_tremula', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Chrysomela_tremulae'] # 25 img ; https://www.gbif.org/occurrence/gallery?taxon_key=4459736
dt_lic_ok[sp == 'Chrysomela_tremula'] # 5 img, but accepted!; https://www.gbif.org/occurrence/gallery?taxon_key=5876339
# These are the same species.
dt_sp_stats[sp == 'Chrysomela_tremulae', ':=' (sp = 'Chrysomela_tremula', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Chrysomela_tremulae', sp := 'Chrysomela_tremula']

dt_gbif_backbone[sp == 'Cryptocephalus_hypochaeridis', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Cryptocephalus_hypochoeridis', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Cryptocephalus_hypochaeridis'] # 7 img # https://www.gbif.org/occurrence/gallery?taxon_key=5876106
dt_lic_ok[sp == 'Cryptocephalus_hypochoeridis'] # 3 img # https://www.gbif.org/occurrence/gallery?taxon_key=6097747
# These are the same species.
dt_sp_stats[sp == 'Cryptocephalus_hypochaeridis', ':=' (sp = 'Cryptocephalus_hypochoeridis', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Cryptocephalus_hypochaeridis', sp := 'Cryptocephalus_hypochoeridis']

dt_gbif_backbone[sp == 'Lochmaea_capreae', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Lochmaea_caprea', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Lochmaea_capreae'] # 237 img https://www.gbif.org/occurrence/gallery?taxon_key=8401929
dt_lic_ok[sp == 'Lochmaea_caprea'] # 65 img https://www.gbif.org/occurrence/gallery?taxon_key=4461358
dt_sp_stats[sp == 'Lochmaea_capreae', ':=' (sp = 'Lochmaea_caprea', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Lochmaea_capreae', sp := 'Lochmaea_caprea']

dt_gbif_backbone[sp == 'Lixus_angustatus', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Lixus_angustus', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Lixus_angustatus'] # 1 img https://www.gbif.org/occurrence/gallery?taxon_key=7371441
dt_lic_ok[sp == 'Lixus_angustus'] # 13 img https://www.gbif.org/occurrence/gallery?taxon_key=1224185
dt_sp_stats[sp == 'Lixus_angustatus', ':=' (sp = 'Lixus_angustus', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Lixus_angustatus', sp := 'Lixus_angustus']


dt_gbif_backbone[sp == 'Psilothrix_viridicoeruleus', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Psilothrix_viridicoerulea', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Psilothrix_viridicoeruleus'] # 56 img https://www.gbif.org/occurrence/gallery?taxon_key=5877582
dt_lic_ok[sp == 'Psilothrix_viridicoerulea'] # 751 img https://www.gbif.org/occurrence/gallery?taxon_key=7942480
dt_sp_stats[sp == 'Psilothrix_viridicoeruleus', ':=' (sp = 'Psilothrix_viridicoerulea', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Psilothrix_viridicoeruleus', sp := 'Psilothrix_viridicoerulea']


dt_gbif_backbone[sp == 'Agrypnus_murina', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Agrypnus_murinus', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Agrypnus_murina'] # 0 img points to Agrypnus murinus
dt_lic_ok[sp == 'Agrypnus_murinus'] # 5842 img https://www.gbif.org/occurrence/gallery?taxon_key=4428942
dt_sp_stats[sp == 'Agrypnus_murina', ':=' (sp = 'Agrypnus_murinus', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Agrypnus_murina', sp := 'Agrypnus_murinus']


dt_gbif_backbone[sp == 'Ampedus_cinnaberinus', .(family, sp, taxonomicStatus)] # all synonyms
dt_gbif_backbone[sp == 'Ampedus_cinnabarinus', .(family, sp, taxonomicStatus)] # accepted
dt_lic_ok[sp == 'Ampedus_cinnaberinus'] # 13 img https://www.gbif.org/occurrence/gallery?taxon_key=4429412
dt_lic_ok[sp == 'Ampedus_cinnabarinus'] # 3 img  https://www.gbif.org/occurrence/gallery?taxon_key=4429415
dt_sp_stats[sp == 'Ampedus_cinnaberinus', ':=' (sp = 'Ampedus_cinnabarinus', taxonomicStatus = 'renamed')]
dt_lic_ok[sp == 'Ampedus_cinnaberinus', sp := 'Ampedus_cinnabarinus']

# Clean list of species to use for sampling
dt_sp_stats[, .N, by = taxonomicStatus]
#    taxonomicStatus    N
# 1:        accepted 1254
# 2:         renamed    9 # drop these, but keep the NAs
# 3:            <NA>   46
# dt_sp_list <- dt_sp_stats[taxonomicStatus != 'renamed'] # This will drop the NAs as well !
dt_sp_list <- dt_sp_stats[taxonomicStatus == 'accepted' | is.na(taxonomicStatus)]
dt_lic_ok[, .(n_url = .N), keyby = .(family, genus, sp)] # also this should have same number of rows as the one above

# {
#   # Attempt to use fuzzy matching for the other cases, but this is risky as they
#   # can be legit species with very similar epithets SO I abandoned this idea.
#   sp_to_check <- dt_sp_stats[is.na(taxonomicStatus), sp]
#   lst_taxa_check <- vector(mode = 'list', length = length(sp_to_check))
#   dt_gbif_coleoptera_sp <- dt_gbif_backbone[order == 'Coleoptera' & taxonRank == "species"]
#   system.time({
#     for (i in 1:length(lst_taxa_check)){
#       spi <- sp_to_check[i]
#       lst_taxa_check[[i]] <- dt_gbif_coleoptera_sp[agrepl(pattern = spi, x = sp, max.distance = 1L), 
#                                                    .(order, family, taxonomicStatus, canonicalName, genericName, specificEpithet, sp)]
#     }
#   })
#   # user  system elapsed 
#   # 89.544   0.199  89.713 
#   
#   names(lst_taxa_check) <- sp_to_check
#   dt_taxa_check <- rbindlist(lst_taxa_check, idcol = "sp_to_check")
#   
#   # Aphodius_conjugatus should be Aphodius coniugatus (which already exists)
#   dt_taxa_check[sp_to_check == 'Aphodius_conjugatus']
#   dt_sp_stats[sp == 'Aphodius_conjugatus', ':=' (sp = 'Aphodius_coniugatus', taxonomicStatus = 'drop')]
#   
#   # Acmaeodera_octodecimguttata
#   dt_taxa_check[sp_to_check == 'Acmaeodera_octodecimguttata']
#   # There is no accepted case, all_synonyms 
#   dt_sp_stats[sp == 'Acmaeodera_octodecimguttata', taxonomicStatus := 'all_synonyms']
#   
#   # Anthaxia_croesa
#   dt_taxa_check[sp_to_check == 'Anthaxia_croesa']
#   # There is no accepted case, all_synonyms 
#   dt_sp_stats[sp == 'Anthaxia_croesa', taxonomicStatus := 'all_synonyms']
#   
#   # Rhagonycha_limbata
#   dt_taxa_check[sp_to_check == 'Rhagonycha_limbata']
#   # There is no accepted case, all_synonyms 
#   dt_sp_stats[sp == 'Rhagonycha_limbata', taxonomicStatus := 'all_synonyms']
#   
#   # Acmaeops_marginata -> Acmaeops_marginatus But if I rename it, then there are
#   # no entries with name Acmaeops_marginatus in the media table...So, do not rename.
#   dt_taxa_check[sp_to_check == 'Acmaeops_marginata']
#   dt_lic_ok[sp == 'Acmaeops_marginatus'] # 0 records for the accepted name
#   dt_lic_ok[sp == 'Acmaeops_marginata'] # 2 records for the synonym name
#   
#   # Similar situation for Acalymma_vittatus, seems that the accepted one is Acalymma_vittatum
#   dt_gbif_coleoptera_sp[sp == 'Acalymma_vittatus']
#   dt_gbif_coleoptera_sp[sp == 'Acalymma_vittatum']
#   dt_lic_ok[sp == 'Acalymma_vittatus']
#   dt_lic_ok[sp == 'Acalymma_vittatum']
# }


# Sample 2022-04-06  ------------------------------------

nrow(dt_sp_list)
# There are now 1300 species. We can sample 2 img per species and will arrive to
# at least 2000 img that can be finally used for training.

spl_lst <- vector(mode = "list", length = nrow(dt_sp_list))
n_img_per_sp <- 2
setkey(dt_lic_ok, sp) # maybe this will speed up a bit
set.seed(2022-04-06)
spl_dt <- dt_lic_ok[, .SD[sample(.N, size = min(2, .N), replace = FALSE)], 
                     keyby = sp]
set.seed(2022-04-06)
spl_dt2 <- dt_lic_ok[, .SD[sample(.N, size = min(2, .N), replace = FALSE)], 
                    keyby = sp]
identical(spl_dt, spl_dt2) # expect TRUE - to be super sure for reproducibility.

# Check if we sampled what we needed. We can get smaller counts per species
# because sometimes there were not enough available images per species (here,
# just 1).
spl_dt[, .N, keyby = sp]
spl_dt[, uniqueN(identifier), keyby = sp] 
# The above should both give same values if URLs are unique
spl_dt[, uniqueN(identifier)] == nrow(spl_dt) # must be TRUE, otherwise you risk to have same URL for multiple rows.

saveRDS(spl_dt, file = "./data/gbif_cleaning/data_processed/coleoptera_sample_2022_04_06.rds")
