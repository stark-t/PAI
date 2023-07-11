# This R script is designed to process and cleanse the GBIF metadata tables
# containing URLs for Diptera. The result is a refined table of metadata and
# URLs, suitable for sampling and downloading images. These images were
# subsequently manually curated and annotated for inclusion in our paper.

library(bit64)
library(data.table)
library(stringr)
library(parallel)

# Read data ---------------------------------------------------------------

# Read GBIF tables. 
# Download link: https://www.gbif.org/occurrence/download/0347148-200613084148143
# See also https://github.com/stark-t/PAI/tree/main/data#readme
# Unzip the folder in ./data/gbif/coleoptera/

# Change nThread to your needs
n_cpu <- detectCores() - 1
dt_occ_all <- fread("./data/gbif_cleaning/gbif/diptera/0347148-200613084148143/occurrence.txt", nThread = n_cpu)
dt_media_all <- fread("./data/gbif_cleaning/gbif/diptera/0347148-200613084148143/multimedia.txt", nThread = n_cpu)
dt_media_all[, media_id := 1:.N]

# Read target taxa - flower visitors and pollinators
# For internal ref - see Dropbox/insect-detection/taxonomy/tk-lists-notes/Diptera.xlsx"
target_taxa_all <- read.csv(file = "./data/gbif_cleaning/taxonomy/diptera.csv",
                            na.strings = c("NA", ""),
                            colClasses = "character")
setDT(target_taxa_all)
target_taxa <- target_taxa_all[, .(superfamily, family)]
# Check if target taxa exists in occurrence data (there might be typos)
target_taxa[, exists_in_occ := family %in% unique(dt_occ_all$family)]

# Check sample ------------------------------------------------------------

# Subset & taxa, lifestage filters ---------------------------------------------

dt_occ <- dt_occ_all[, .(gbifID, order, family, genus, specificEpithet, species, 
                         lifeStage, sex,
                         continent, countryCode, level0Name, level0Gid,
                         publisher, collectionID, datasetName, basisOfRecord, year, samplingProtocol,
                         identifiedBy, nomenclaturalStatus, repatriated)]
# Only continent, countryCode were used for location filtering; level0Name,
# level0Gid have more NA-s than countryCode.

rm(dt_occ_all); gc()

# Selected only target taxa as defined by Tiffany
dt_occ <- dt_occ[family %in% target_taxa$family]

# Discard unwanted life stages. Keep the unknowns also.
dt_occ[, .N, by = lifeStage][order(-N)]
dt_occ <- dt_occ[lifeStage %in% c("Imago", "Adult", "Unknown" , "")]
dt_occ[, .N, by = lifeStage][order(-N)]
# 1:           227167
# 2:     Imago 100382
# 3:     Adult  70002
# 4:   Unknown  31512

# Discard unidentified genera and species
dt_occ[genus == "", .N] # 14534
dt_occ[is.na(genus), .N] # 0
dt_occ <- dt_occ[genus != ""]

dt_occ[specificEpithet == "", .N] # 50900
dt_occ[is.na(specificEpithet), .N] # 0
dt_occ <- dt_occ[specificEpithet != ""]

dt_occ[species == "", .N] # 4
dt_occ[species == ""]
dt_occ <- dt_occ[species != ""]

# Merge with media --------------------------------------------------------

# For each gbifID in the occurrence table there can be more entries in the media
# table. So, gbifID is unique only in the occurrence table.
length(unique(dt_occ$gbifID)) == nrow(dt_occ) # expect TRUE
length(unique(dt_media_all$gbifID)) == nrow(dt_media_all) # expect FALSE

dt_occ_media <- merge(dt_occ, dt_media_all, by = "gbifID")
length(unique(dt_occ_media$gbifID)) == nrow(dt_occ_media) # expect FALSE
length(unique(dt_occ_media$media_id)) == nrow(dt_occ_media) # expect TRUE
sum(duplicated(dt_occ_media$media_id)) # expect 0

rm(dt_media_all); gc()

# Data quality type filters -----------------------------------------------

# ~ basisOfRecord ---------------------------------------------------------

# Keep only HUMAN_OBSERVATION
dt_occ_media[,.N, by = basisOfRecord][order(-N)]
#         basisOfRecord       N
# 1:   HUMAN_OBSERVATION 556146
# 2:  PRESERVED_SPECIMEN  44706
# 3:     MATERIAL_SAMPLE   1102
# 4:     FOSSIL_SPECIMEN    323
# 5:             UNKNOWN    114
# 6: MACHINE_OBSERVATION      3
dt_occ_media <- dt_occ_media[basisOfRecord == "HUMAN_OBSERVATION"]


# ~ type ------------------------------------------------------------------

# Keep only StillImage
dt_occ_media[, .N, by = type]
# 1: StillImage 556141
# 2:      Sound      5
dt_occ_media <- dt_occ_media[type == "StillImage"]


# ~ file format -----------------------------------------------------------

# Keep only image/jpeg
dt_occ_media[, .N, by = format][order(-N)]
# 1:                   image/jpeg 553571
# 2:                    image/png   2403 # this might have potential too!
# 3: tiff, jpeg, jpeg, jpeg, jpeg     85
# 4:                    image/gif     43
# 5:                  image/pjpeg     22
# 6:                   image/scan     17
dt_occ_media <- dt_occ_media[format %in% c("image/jpeg", "image/png")]


# Check file format by looking at the URL. It turns out this is not that useful.
dt_occ_media[, is_jpg := grepl(pattern = ".jpg", x = identifier, fixed = TRUE)]
dt_occ_media[, .N, by = is_jpg][order(-N)]
#    is_jpg      N
# 1:   TRUE 377811
# 2:  FALSE 178163
dt_occ_media[is_jpg == FALSE]
# There is also .jpeg and also some URLS seem valid even though they do not have
# the jpg keyword.
dt_occ_media[, is_jpg := NULL]


# ~ samplingProtocol ------------------------------------------------------

# samplingProtocol brings some useful info. Keep only "" and "seen".
dt_occ_media[, .N, by = samplingProtocol][order(-N)]
#                samplingProtocol      N
# 1:                              509014
# 2:                         seen  42777
# 3:  caught by hand and released   1338
# 4:      microscopic examination   1264
# 5: caught by hand and collected    733
# 6:                      indoors    240
# 7:       specimen in collection    222
# 8:                    lighttrap    204
# 9:                     at sugar     41
# 10:                   other trap     40
# 11:                   colourtrap     33
# 12:            genitals examined     29
# 13:           grown and released     20
# 14:                  malaisetrap      9
# 15:                 pitfall trap      8
# 16:                       tracks      2

# I checked randomly some of the other protocols and they do not seem to have
# insects on flowers
dt_occ_media[samplingProtocol == "indoors"]

dt_occ_media <- dt_occ_media[samplingProtocol %in% c("", "seen")]


# ~ collectionID ----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter. Also has too many NAs (doesn't bring useful info).
dt_occ_media[, .N, by = collectionID]
#                                      collectionID      N
# 1:                                                534641
# 2: https://bison.usgs.gov/ipt/resource?r=bugguide  17150
dt_occ_media[collectionID == "https://bison.usgs.gov/ipt/resource?r=bugguide"]
# Some images seem to be of insects on flowers, e.g.:
# https://bugguide.net/images/cache/JZK/RCZ/JZKRCZSR0H3LOLKZTLQZDLIZALSZOLXZ9L7RDZKRYZ0RDZRZ3ZXRFZZZ9LYLVL6RHHPRJZJLYZERRHGRVL6RZHPRKHXZ.jpg


# ~ repatriated -----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter
dt_occ_media[,.N, by = repatriated][order(-N)]
# 1:       FALSE 353395
# 2:        TRUE 197896
# 3:          NA    500
dt_occ_media[repatriated == TRUE]


# ~ datasetName -----------------------------------------------------------

# Also, this doesn't help much to identify insects from collections at this point.

dt_occ_media[,.N, by = datasetName][order(-N)]
#                                                                                                                  datasetName      N
# 1:                                                                                   iNaturalist Research-grade Observations 337261
# 2:                                                                        Observation.org, Nature data from around the World 160415
# 3:                                                                                     Norwegian Species Observation Service  31996
# 4:          BugGuide - Identification, Images, & Information For Insects, Spiders & Their Kin For the United States & Canada  17150
# 5:                                                                                               Earth Guardians Weekly Feed   1759
# 6:                                                                                                       Canberra Nature Map   1014
# 7:                                                                              Biodiversity4all Research-Grade Observations    661
# 8:                                                                 Taiwan Moth Occurrence Data Collected From Social Network    349
# 9:                                                               Species recordings from the Danish National portal Arter.dk    343
# 10:                                                                                         ALA species sightings and OzAtlas    342
# 11:                                                                                             Estonian Naturalists’ Society    242
# 12:                                                                                   Atlas of Life in the Coastal Wilderness    120
# 13:                                                                                                               NatureShare     60
# 14:                                                                                                                     PaDIL     40
# 15:                                                                       India Biodiversity Portal publication grade dataset     25
# 16: Images and observations of mostly edible plants in Stephen Barstow’s Edible Garden in Norway, taken between 2005 and 2014     10
# 17:                                                                                              Australian Museum Factsheets      2
# 18:                                                           Images of Flora and Fauna of 290 Leppitt Rd, Upper Beaconsfield      1
# 19:                                                                                                                Gaia Guide      1


# ~ publisher -------------------------------------------------------------

dt_occ_media[,.N, by = publisher.x][order(-N)] # publisher.x is from occurrence file
dt_occ_media[,.N, by = publisher.y][order(-N)] # publisher.y is from media file

# ~ year ------------------------------------------------------------------

# Be aware of the NA-s, they are valid. I didn't set yet a filter here. The old
# ones are really a few though.
dt_occ_media[, .N, by = year][order(-N)]
dt_occ_media <- dt_occ_media[year >= 1990 | is.na(year)]

# ~ identifier ------------------------------------------------------------

# Check if identifier is missing
dt_occ_media[identifier == "", .N] # 0
dt_occ_media[is.na(identifier), .N] # 0

# Check if identifier is unique - not the case!
length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE, but gave FALSE
# It seems that 28 iNaturalist records have a duplicate.
dt_occ_media[duplicated(identifier)]
dt_occ_media <- dt_occ_media[! duplicated(identifier)]

length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE now

# ~ nomenclaturalStatus ---------------------------------------------------

# nomenclaturalStatus doesn't help to filter - all NA
dt_occ_media[, .N, by = nomenclaturalStatus][order(-N)]
#    nomenclaturalStatus      N
# 1:                  NA 551618


# ~ identifiedBy ----------------------------------------------------------

# A lot fo them are unknown
dt_occ_media[, .N, by = identifiedBy][order(-N)]
#       identifiedBy      N
# 1:                 211582
# 2:  Even Dankowicz  20636
# 3:   Trina Roberts  18182
# 4:   Michael Knapp  12007
# 5: Caleb Scholtens   6957
# ---                       
# 22425: Martin Sørensen      1
# 22426:  Anders Tøttrup      1
# 22427:      Ole Olesen      1
# 22428:    Karin Jensen      1
# 22429:  Nikolaj Hansen      1


# Media license -----------------------------------------------------------

# Check licenses
dt_lic <- dt_occ_media[, .N, by = license][order(-N)]
dt_lic
#                                              license      N
# 1:    http://creativecommons.org/licenses/by-nc/4.0/ 287994
# 2: http://creativecommons.org/licenses/by-nc-nd/4.0/  91646
# 3:       http://creativecommons.org/licenses/by/4.0/  55383
# 4:                               All rights reserved  39064
# 5:                             © All rights reserved  29260
# 6:                                                    21255
# 7: http://creativecommons.org/licenses/by-nc-sa/4.0/  12295
# 8: http://creativecommons.org/publicdomain/zero/1.0/   7859
# 9:    http://creativecommons.org/licenses/by-sa/4.0/   5686
# 10:                                      CC BY-ND 4.0   1070
# 11:    http://creativecommons.org/licenses/by-nd/4.0/    101
# 12:                                 Australian Museum      2
# 13:          Creative Commons Attribution-Share Alike      1
# 14:                                                ??      1
# 15:       http://creativecommons.org/licenses/by/3.0/      1

# saveRDS(dt_lic, file = "data_processed/dt_diptera_licenses.rds")

# Check https://creativecommons.org/faq/#artificial-intelligence-and-cc-licenses

# Permissible are all creativecommons ones from above including NoDerivatives:

# The NoDerivatives (nd) part apply because:
# https://creativecommons.org/faq/#When_is_my_use_considered_an_adaptation.3F

# Mark licensees about which we are sure and not sure.
license_unsure <- c("All rights reserved", 
                    "© All rights reserved", 
                    "",
                    "Usage Conditions Apply", # this appeared at some point, but the filters above kicked it out
                    "Australian Museum", 
                    "??")
dt_occ_media[, license_ok := ! license %in% license_unsure]


# Location filters --------------------------------------------------------

# A lot of them do not have continent info!
dt_occ_media[, .N, by = continent][order(-N)]
#        continent      N
# 1:               390549
# 2:        EUROPE 160897
# 3:          ASIA     81
# 4:        AFRICA     66
# 5: NORTH_AMERICA     21
# 6: SOUTH_AMERICA      4

# But for some we can get the country code
dt_occ_media[continent == ""][countryCode == "", .N] # 500
dt_occ_media[continent == ""][level0Gid == "", .N] # 20200
dt_occ_media[continent == ""][level0Name == "", .N] # 20200
# The countryCode seems to be more comprehensive than level0Gid or level0Name
dt_occ_media[continent == ""][level0Gid == ""][countryCode == "", .N] # 500

dt_occ_media[countryCode == "", level0Gid]
dt_occ_media[level0Gid == "", countryCode]

# Read country code data
country <- fread('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')
# Or see in ./data/gbif_cleaning/cache/country_codes.csv

unique(country$region)
country[region == ""]
europe <- country[region == "Europe", "alpha-2"][[1]]

# Update EUROPE info based on country code. Not many are gained though
dt_occ_media[countryCode %in% europe, continent := "EUROPE"]
dt_occ_media[continent == "", continent := "no_info"]
dt_occ_media[, .N, by = continent][order(-N)]
#        continent      N
# 1:        EUROPE 315077
# 2:       no_info 236370
# 3:          ASIA     80
# 4:        AFRICA     66
# 5: NORTH_AMERICA     21
# 6: SOUTH_AMERICA      4

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
#    european      N
# 1:     TRUE 383443
# 2:       NA 168175

# Also, a lot of those without continent info are in US and CA.
dt_occ_media[is.na(european)][countryCode %in% europe] # expect 0 rows
dt_occ_media[is.na(european)][! countryCode %in% europe][, .N, by = countryCode][order(-N)]

# Use the lists for Syprhid that are marked to be in Europe in order to salvage
# URLs that can be European also. 
# The info was extracted from:
# Speight, M.C.D. (2017) Species accounts of European Syrphidae, 2017. Syrph the Net, the database of
# European Syrphidae (Diptera), vol. 97, 294 pp., Syrph the Net publications, Dublin.
syrphids <- fread("./data/gbif_cleaning/taxonomy/syrphidae_range_hiwis.csv")
syrphids[,.N, by = rare]
syrphids_europe <- syrphids[rare %in% c("no", "unclear")]
# Also, not in Europe: Chrysotoxum persicum, Copestylum melleum
syrphids_europe <- syrphids_europe[! species %in% c("Chrysotoxum persicum", "Copestylum melleum")]

# If 0 rows, then all were marked as being also in Europe already
dt_occ_media[is.na(european)][genus_epithet %in% syrphids_europe$species]


# Save file ---------------------------------------------------------------

# Save only a subset of columns for the curated url-s
dt_diptera_eu_clean_url <- dt_occ_media[european == TRUE, 
                                .(media_id, gbifID, order, family, genus, specificEpithet, 
                                  lifeStage, sex, countryCode,
                                  identifier, license_ok)]

saveRDS(object = dt_diptera_eu_clean_url, 
        file = "./data/gbif_cleaning/data_processed/dt_diptera_eu_clean_url.rds")
