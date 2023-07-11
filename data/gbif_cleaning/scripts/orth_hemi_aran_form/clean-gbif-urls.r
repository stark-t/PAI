# This R script is designed to process and cleanse the GBIF metadata tables
# containing URLs for orthoptera, hemiptera, araneae, formicidae. The result is
# a refined table of metadata and URLs, suitable for sampling and downloading
# images. These images were subsequently manually curated and annotated for
# inclusion in our paper.

# Orthoptera (grasshoppers and crikets)
# Hemiptera (true bugs)
# Araneae (spiders)
# Ants (the Formicidae family within Hymenoptera)

library(bit64)
library(data.table)
library(stringr)
library(parallel)

# Read data ---------------------------------------------------------------

# Read GBIF tables. 
# Download links: 
# - orthoptera https://www.gbif.org/occurrence/download/0006939-210914110416597
#   Unzip the folder in ./data/gbif/orthoptera/
# - hemiptera https://www.gbif.org/occurrence/download/0033945-210914110416597
#   Unzip the folder in ./data/gbif/hemiptera/
# - araneae  https://www.gbif.org/occurrence/download/0033954-210914110416597
#   Unzip the folder in ./data/gbif/araneae/
# - formicidae https://www.gbif.org/occurrence/download/0033980-210914110416597
#   Unzip the folder in ./data/gbif/formicidae/

# See also https://github.com/stark-t/PAI/tree/main/data#readme

# Change nThread to your needs
n_cpu <- detectCores() - 1

dt_occ_all_orthoptera <- fread("data/orthoptera/0006939-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all_orthoptera <- fread("data/orthoptera/0006939-210914110416597/multimedia.txt", nThread = n_cpu)

dt_occ_all_hemiptera <- fread("data/hemiptera/0033945-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all_hemiptera <- fread("data/hemiptera/0033945-210914110416597/multimedia.txt", nThread = n_cpu)

dt_occ_all_araneae <- fread("data/araneae/0033954-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all_araneae <- fread("data/araneae/0033954-210914110416597/multimedia.txt", nThread = n_cpu)

dt_occ_all_formicidae <- fread("data/formicidae/0033980-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all_formicidae <- fread("data/formicidae/0033980-210914110416597/multimedia.txt", nThread = n_cpu)

# Merge all media together. I merged all occurrence tables later due to errors
# (see below). I named the list by order so that I can use it as a key for
# merging later with occurrence tables.
dt_media_all <- rbindlist(list(Orthoptera = dt_media_all_orthoptera,
                               Hemiptera = dt_media_all_hemiptera,
                               Araneae = dt_media_all_araneae,
                               Hymenoptera = dt_media_all_formicidae), 
                          idcol = "order")
dt_media_all[, media_id := 1:.N]

rm(dt_media_all_orthoptera, dt_media_all_hemiptera, dt_media_all_araneae, dt_media_all_formicidae)
gc()

# Subset & taxa, lifestage filters ---------------------------------------------

# Only continent, countryCode were used for location filtering; level0Name,
# level0Gid have more NA-s than countryCode.
dt_occ_orthoptera <- dt_occ_all_orthoptera[, .(gbifID, order, family, genus, specificEpithet, species, 
                                               lifeStage, sex,
                                               continent, countryCode, level0Name, level0Gid,
                                               publisher, collectionID, datasetName, basisOfRecord, year, samplingProtocol,
                                               identifiedBy, nomenclaturalStatus, repatriated)]
dt_occ_hemiptera <- dt_occ_all_hemiptera[, .(gbifID, order, family, genus, specificEpithet, species, 
                                             lifeStage, sex,
                                             continent, countryCode, level0Name, level0Gid,
                                             publisher, collectionID, datasetName, basisOfRecord, year, samplingProtocol,
                                             identifiedBy, nomenclaturalStatus, repatriated)]
dt_occ_araneae <- dt_occ_all_araneae[, .(gbifID, order, family, genus, specificEpithet, species, 
                                         lifeStage, sex,
                                         continent, countryCode, level0Name, level0Gid,
                                         publisher, collectionID, datasetName, basisOfRecord, year, samplingProtocol,
                                         identifiedBy, nomenclaturalStatus, repatriated)]
dt_occ_formicidae <- dt_occ_all_formicidae[, .(gbifID, order, family, genus, specificEpithet, species, 
                                               lifeStage, sex,
                                               continent, countryCode, level0Name, level0Gid,
                                               publisher, collectionID, datasetName, basisOfRecord, year, samplingProtocol,
                                               identifiedBy, nomenclaturalStatus, repatriated)]
# Bind occurrence tables together. I had to do it here, because in the full
# tables (all columns) some columns are read differently. For example I got
# error "Class attribute on column 144 of item 2 does not match with column 144
# of item 1" when I tried the same thing on raw data (all columns).
dt_occ <- rbindlist(list(dt_occ_orthoptera,
                         dt_occ_hemiptera,
                         dt_occ_araneae,
                         dt_occ_formicidae))

rm(dt_occ_all_orthoptera, dt_occ_all_hemiptera, dt_occ_all_araneae, dt_occ_all_formicidae)
rm(dt_occ_orthoptera, dt_occ_hemiptera, dt_occ_araneae, dt_occ_formicidae)
gc()

# Discard unwanted life stages. Keep the unknowns also.
dt_occ[, .N, by = lifeStage][order(-N)]
dt_occ <- dt_occ[lifeStage %in% c("Imago", "Adult", "Unknown" , "")]
dt_occ[, .N, by = lifeStage][order(-N)]
#    lifeStage       N
# 1:           1080149
# 2:     Adult  346034
# 3:     Imago  164967
# 4:   Unknown   69380

# Discard unidentified genera and species
dt_occ[genus == "", .N] # 22598
dt_occ[is.na(genus), .N] # 0
dt_occ <- dt_occ[genus != ""]

dt_occ[specificEpithet == "", .N] # 61037
dt_occ[is.na(specificEpithet), .N] # 0
dt_occ <- dt_occ[specificEpithet != ""]

dt_occ[species == "", .N] # 14
dt_occ[species == ""]
dt_occ <- dt_occ[species != ""]

# Merge with media --------------------------------------------------------

# For each gbifID in the occurrence table there can be more entries in the media
# table. So, gbifID is unique only in the occurrence table. Also, not all
# occurrences will have URLs associated with them, so is ok to lose rows after
# merging.
dt_occ_media <- merge(dt_occ, dt_media_all, by = c("gbifID", "order"))
length(unique(dt_occ_media$media_id)) == nrow(dt_occ_media) # expect TRUE
sum(duplicated(dt_occ_media$media_id)) # expect 0

rm(dt_media_all); gc()

# Data quality type filters -----------------------------------------------

# ~ basisOfRecord ---------------------------------------------------------

# Keep only HUMAN_OBSERVATION - already done at download time from GBIF
dt_occ_media[,.N, by = basisOfRecord][order(-N)]
#          basisOfRecord       N
# 1:   HUMAN_OBSERVATION 2589536


# ~ type ------------------------------------------------------------------

# Keep only StillImage
dt_occ_media[, .N, by = type]
# 1:  StillImage 2587704
# 2:                 212
# 3: MovingImage      17
# 4:       Sound    1603
dt_occ_media <- dt_occ_media[type == "StillImage"]


# ~ file format -----------------------------------------------------------

# Keep only image/jpeg
dt_occ_media[, .N, by = format][order(-N)]
#         format       N
# 1:  image/jpeg 2576802
# 2:   image/png   10304
# 3:   image/gif     275
# 4: image/pjpeg     163
# 5:  image/scan     156
# 6:  image/tiff       4
dt_occ_media[format == "image/png", .(identifier)]
dt_occ_media <- dt_occ_media[format %in% c("image/jpeg", "image/png")]

# Note that if you try to look if the .jpg or .jpeg keywords, sometimes they are
# missing, but the urls are valid (see the case for Diptera).


# ~ samplingProtocol ------------------------------------------------------

# samplingProtocol brings some useful info. Keep only "" and "seen".
dt_occ_media[, .N, by = samplingProtocol][order(-N)]
#                                                     samplingProtocol       N
# 1:                                                                   2482503
# 2:                                                              seen   84303
# 3:                                                             trawl    5340
# 4:                                                       beating net    4572
# 5:                                       caught by hand and released    2104
# 6:                                                         lighttrap    1849
# 7:                                                           indoors    1330
# 8:                                                    caught by hand    1326
# 9:                                                    seen and heard     611
# 10:                                                       net by hand     529
# 11:                                      caught by hand and collected     408
# 12:                                                            caught     399
# 13:                                                        other trap     374
# 14:                                                    beating screen     359
# 15:                                                      pitfall trap     342
# 16:                                                          at sugar     273
# 17:                                           microscopic examination     207
# 18:                                            specimen in collection      85
# 19:                                                             heard      77
# 20:                                                          on sheet      30
# 21:                                        observation in the daytime      12
# 22:                                                         collected      12
# 23:                                                       malaisetrap      12
# 24:                                                        colourtrap      11
# 25:                                                grown and released       8
# 26:                                                  with batdetector       8
# 27:                                          observation in the night       7
# 28:                                           grown and in collection       5
# 29:                                                            tracks       3
# 30:         handnet; photos taken in container with scale/color chart       2
# 31:                                                            Manual       2
# 32: handnet; photos taken on hand using scale/color chart finger ring       1
# 33:                                        berlese funnel soil sample       1
# 34:                                                 field observation       1


# The "trawl" ones seem to be in lab conditions, or close up photos.
test <- dt_occ_media[samplingProtocol == "trawl"]
test[, .N, by = identifier]

# The "beating net" ones seem to be in lab conditions, or close up photos.
test <- dt_occ_media[samplingProtocol == "beating net"]
test[, .N, by = identifier]

dt_occ_media <- dt_occ_media[samplingProtocol %in% c("", "seen")]


# ~ collectionID ----------------------------------------------------------

# Has too many NAs (doesn't bring useful info).
dt_occ_media[, .N, by = collectionID]
#                                       collectionID      N
# 1:                                                2477884
# 2:                                             50     255
# 3:                http://grbio.org/cool/1wcv-afpg     872
# 4: https://bison.usgs.gov/ipt/resource?r=bugguide   87795


# ~ repatriated -----------------------------------------------------------

# This is also not a good filter
dt_occ_media[,.N, by = repatriated][order(-N)]
# 1:       FALSE 1419322
# 2:        TRUE 1142614
# 3:          NA    4870
dt_occ_media[repatriated == TRUE, identifier]


# ~ datasetName -----------------------------------------------------------

# Also, this doesn't help much to identify insects from collections at this point.
dt_occ_media[,.N, by = datasetName][order(-N)]
#                                                                                                                  datasetName       N
# 1:                                                                                   iNaturalist Research-grade Observations 2092468
# 2:                                                                        Observation.org, Nature data from around the World  289080
# 3:          BugGuide - Identification, Images, & Information For Insects, Spiders & Their Kin For the United States & Canada   87795
# 4:                                                                                               Earth Guardians Weekly Feed   39031
# 5:                                                                                     Norwegian Species Observation Service   31868
# 6:                                                                                                       Canberra Nature Map    8882
# 7:                                                                                         ALA species sightings and OzAtlas    3441
# 8:                                                                          ConDidact Citizen Science Surveys - Spiders 2014    2841
# 9:                                                                              Biodiversity4all Research-Grade Observations    2805
# 10:                                                               Species recordings from the Danish National portal Arter.dk    2329
# 11:                                                                                   Atlas of Life in the Coastal Wilderness    1290
# 12:                                                                          ConDidact Citizen Science Surveys - Spiders 2011    1154
# 13:                                                                       India Biodiversity Portal publication grade dataset     877
# 14:                                                                                              NMNH Extant Specimen Records     872
# 15:                                                                                             Estonian Naturalists’ Society     728
# 16:                                                                                                               NatureShare     450
# 17:                                                                                          UAM Insect Observations (Arctos)     255
# 18:                                                                            Xeno-canto - Bird sounds from around the world     139
# 19:                                                                                                                     PaDIL     133
# 20:                                                           Images of Flora and Fauna of 290 Leppitt Rd, Upper Beaconsfield     101
# 21: Images and observations of mostly edible plants in Stephen Barstow’s Edible Garden in Norway, taken between 2005 and 2014      69
# 22:                                                                              Collections of Bioclass, school #179, Moscow      62
# 23:         RSCROP: Desert Locust Monitoring, Forecasting and Assessment in Africa and Asia Archive from 2018-01-01 (Ongoing)      24
# 24:                                                                                             IBF Monitoring of Plant Galls      21
# 25:                                                                                             Michael Barkla's Observations      18
# 26:                                                                                              IBF Monitoring of Orthoptera      16
# 27:                                                                                             Southern Highlands Nature Map      12
# 28:                                                               My naturesounds - nature observations with sound recordings      11
# 29:                                                         Observations of the emergence of Fijian cicadas in September 2017       9
# 30:                                                                                                              ClimateWatch       8
# 31:                                                                                           Newhaven Sanctuary observations       6
# 32:                                                                                                 Albury Wodonga Nature Map       5
# 33:                                                                                                               Auswildlife       2
# 34:                                                                                  City of Kalamunda Biodiversity Inventory       2
# 35:                                                                                                                Gaia Guide       1
# 36:                                                                                                    Noosa Shire Nature Map       1


# ~ publisher -------------------------------------------------------------

dt_occ_media[,.N, by = publisher.x][order(-N)] # publisher.x is from occurrence file
dt_occ_media[,.N, by = publisher.y][order(-N)] # publisher.y is from media file


# ~ year ------------------------------------------------------------------

# Be aware of the NA-s, they are valid. The old ones are really a few though.
dt_occ_media[, .N, by = year][order(-N)]
dt_occ_media <- dt_occ_media[year >= 1990 | is.na(year)]


# ~ identifier ------------------------------------------------------------

# Check if identifier is missing
dt_occ_media[identifier == "", .N] # 0
dt_occ_media[is.na(identifier), .N] # 0

# Check if identifier is unique - not the case!
length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE, but gave FALSE
# It seems that 597 records have a duplicate.
dt_occ_media[duplicated(identifier), .N, by = publisher.x]
dt_occ_media <- dt_occ_media[! duplicated(identifier)]

length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE now


# ~ nomenclaturalStatus ---------------------------------------------------

# nomenclaturalStatus doesn't help to filter - all NA
dt_occ_media[, .N, by = nomenclaturalStatus][order(-N)]
#    nomenclaturalStatus      N
# 1:                  NA 2565158

# ~ identifiedBy ----------------------------------------------------------

# A lot fo them are unknown
dt_occ_media[, .N, by = identifiedBy][order(-N)]


# Media license -----------------------------------------------------------

# Check licenses
dt_lic <- dt_occ_media[, .N, by = license][order(-N)]
dt_lic
#                                               license      N
# 1:    http://creativecommons.org/licenses/by-nc/4.0/ 1771719
# 2:       http://creativecommons.org/licenses/by/4.0/  235864
# 3: http://creativecommons.org/licenses/by-nc-nd/4.0/  184414
# 4:                                                    145543
# 5:                               All rights reserved   76771
# 6: http://creativecommons.org/publicdomain/zero/1.0/   50966
# 7: http://creativecommons.org/licenses/by-nc-sa/4.0/   41007
# 8:    http://creativecommons.org/licenses/by-sa/4.0/   29240
# 9:                             © All rights reserved   27522
# 10:                            Usage Conditions Apply     867
# 11:                                      CC BY-ND 4.0     673
# 12:    http://creativecommons.org/licenses/by-nd/4.0/     526
# 13:          Creative Commons Attribution-Share Alike      30
# 14:                                        http://...       6
# 15:                  Creative Commons Attribution 3.0       4
# 16:                                                ??       3
# 17:  Photo: Bruce Thomson, http://www.auswildlife.com       2
# 18:       http://creativecommons.org/licenses/by/3.0/       1

# Mark licensees about which we are sure and not sure.
license_unsure <- c("",
                    "All rights reserved", 
                    "© All rights reserved", 
                    "Usage Conditions Apply",
                    "http://...",
                    "??",
                    "Photo: Bruce Thomson, http://www.auswildlife.com")
dt_occ_media[, license_ok := ! license %in% license_unsure]
dt_occ_media[license_ok == TRUE, .N, by = license][order(-N)] 
dt_occ_media[license_ok == FALSE, .N, by = license][order(-N)] 


# Location filters --------------------------------------------------------

# A lot of them do not have continent info!
dt_occ_media[, .N, by = continent][order(-N)]
#         continent      N
# 1:               2269260
# 2:        EUROPE  295193
# 3: NORTH_AMERICA     357
# 4:          ASIA     144
# 5:        AFRICA     137
# 6: SOUTH_AMERICA      61
# 7:       OCEANIA       6

# But for some we can get the country code
dt_occ_media[continent == ""][countryCode == "", .N] # 4841
dt_occ_media[continent == ""][level0Gid == "", .N] # 112528
dt_occ_media[continent == ""][level0Name == "", .N] # 112528
# The countryCode seems to be more comprehensive than level0Gid or level0Name
dt_occ_media[continent == ""][level0Gid == ""][countryCode == "", .N] # 4841

dt_occ_media[countryCode == "", level0Gid]
dt_occ_media[level0Gid == "", countryCode]

# Read country code data
country <- fread('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')
# Or see in ./data/gbif_cleaning/cache/country_codes.csv

unique(country$region)
country[region == ""]
europe <- country[region == "Europe", "alpha-2"][[1]]

# Update EUROPE info based on country code.
dt_occ_media[countryCode %in% europe, continent := "EUROPE"]
dt_occ_media[continent == "", continent := "no_info"]
dt_occ_media[, .N, by = continent][order(-N)]
#        continent       N
# 1:       no_info 1735979
# 2:        EUROPE  828475
# 3: NORTH_AMERICA     357
# 4:          ASIA     143
# 5:        AFRICA     137
# 6: SOUTH_AMERICA      61
# 7:       OCEANIA       6

# We lose images of insects that can be both in Europe and other continents. A
# lot of them can be in US or CA and also appear in Europe, but are not marked
# in Europe. Salvage some:

# Find species that are both in Europe and other continents. We can salvage URLs
# from other continents too if they appear in Europe.
dt_occ_media[, genus_epithet := paste(genus, specificEpithet)]
sp_cont_info <- dt_occ_media[, .( continents = paste(unique(continent), collapse = ", ") ), 
                             by = genus_epithet]
sp_cont_info[, in_europe := grepl(pattern = "EUROPE", x = continents)]
european_sp <- sp_cont_info[in_europe == TRUE, genus_epithet]
dt_occ_media[genus_epithet %in% european_sp, european := TRUE]
dt_occ_media[, .N, by = european][order(-N)]
#     european      N
# 1:       NA 1423480
# 2:     TRUE 1141678

# Also, a lot of those without continent info are in US and CA.
dt_occ_media[is.na(european)][countryCode %in% europe] # expect 0 rows
dt_occ_media[is.na(european)][! countryCode %in% europe][, .N, by = countryCode][order(-N)]
#    countryCode      N
# 1:          US 854935
# 2:          AU 152268
# 3:          CA  74852
# 4:          MX  50770
# 5:          NZ  36883


# Save file ---------------------------------------------------------------

# Save only a subset of columns for the curated url-s
dt_eu_clean_url <- dt_occ_media[european == TRUE, 
                                .(media_id, gbifID, order, family, genus, specificEpithet, 
                                  lifeStage, sex, countryCode,
                                  identifier, license_ok)]

saveRDS(object = dt_eu_clean_url, 
        file = "./data/gbif_cleaning/data_processed/dt_orth_hemi_aran_form_eu_clean_url.rds")
