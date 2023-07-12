# This R script is designed to process and cleanse the GBIF metadata tables
# containing URLs for Hymenoptera (no ants). The result is a refined table of
# metadata and URLs, suitable for sampling and downloading images. These images
# were subsequently manually curated and annotated for inclusion in our paper.

library(bit64)
library(data.table)
library(stringr)
library(parallel)

# Read data ---------------------------------------------------------------

# Read GBIF tables. 
# Download link: https://www.gbif.org/occurrence/download/0002743-210914110416597
# See also https://github.com/stark-t/PAI/tree/main/data#readme
# Unzip the folder in ./data/gbif/hymenoptera/

# Change nThread to your needs
n_cpu <- detectCores() - 1
dt_occ_all <- fread("./data/gbif_cleaning/gbif/hymenoptera/0002743-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all <- fread("./data/gbif_cleaning/gbif/hymenoptera/0002743-210914110416597/multimedia.txt", nThread = n_cpu)
dt_media_all[, media_id := 1:.N]

# Read target taxa - flower visitors and pollinators
# For internal ref - see Dropbox/insect-detection/taxonomy/tk-lists-notes/Hymenoptera.xlsx"
target_taxa_all <- read.csv(file = "./data/gbif_cleaning/taxonomy/hymenoptera.csv",
                            na.strings = c("NA", ""),
                            colClasses = "character")
setDT(target_taxa_all)
target_taxa <- target_taxa_all[, .(family)]
# Check if target taxa exists in occurrence data (there might be typos)
target_taxa[, exists_in_occ := family %in% unique(dt_occ_all$family)]
target_taxa[exists_in_occ == FALSE, family]

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
# 1:           559272
# 2:     Adult 509809
# 3:     Imago  50927
# 4:   Unknown  16661

# Discard unidentified genera and species
dt_occ[genus == "", .N] # 16288
dt_occ[is.na(genus), .N] # 0
dt_occ <- dt_occ[genus != ""]

dt_occ[specificEpithet == "", .N] # 169684
dt_occ[is.na(specificEpithet), .N] # 0
dt_occ <- dt_occ[specificEpithet != ""]

dt_occ[species == "", .N] # 5
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
#          basisOfRecord       N
# 1:   HUMAN_OBSERVATION 1122166
# 2:  PRESERVED_SPECIMEN  451561
# 3:     MATERIAL_SAMPLE    6416
# 4:     FOSSIL_SPECIMEN     792
# 5:             UNKNOWN     295
# 6: MACHINE_OBSERVATION      26
dt_occ_media <- dt_occ_media[basisOfRecord == "HUMAN_OBSERVATION"]


# ~ type ------------------------------------------------------------------

# Keep only StillImage
dt_occ_media[, .N, by = type]
# 1:  StillImage 1122126
# 2: MovingImage       3
# 3:                   1
# 4:       Sound      36
dt_occ_media <- dt_occ_media[type == "StillImage"]


# ~ file format -----------------------------------------------------------

# Keep only image/jpeg
dt_occ_media[, .N, by = format][order(-N)]
#                          format       N
# 1:                   image/jpeg 1116902
# 2:                    image/png    4958 # this might have potential too!
# 3:                    image/gif      91
# 4: tiff, jpeg, jpeg, jpeg, jpeg      82
# 5:                   image/scan      48
# 6:                  image/pjpeg      45
dt_occ_media[format == "image/png", .(identifier)]
dt_occ_media <- dt_occ_media[format %in% c("image/jpeg", "image/png")]

# Note that if you try to look if the .jpg or .jpeg keywords, sometimes they are
# missing, but the urls are valid (see the case for Diptera).


# ~ samplingProtocol ------------------------------------------------------

# samplingProtocol brings some useful info. Keep only "" and "seen".
dt_occ_media[, .N, by = samplingProtocol][order(-N)]
#                samplingProtocol       N
# 1:                              1085506
# 2:                         seen   33511
# 3:  caught by hand and released    1755
# 4: caught by hand and collected     537
# 5:                    lighttrap     149
# 6:                      indoors     126
# 7:       specimen in collection     105
# 8:                   colourtrap      46
# 9:                   other trap      35
# 10:                     at sugar      33
# 11:                 pitfall trap      14
# 12:      microscopic examination      12
# 13:      grown and in collection       9
# 14:                       tracks       8
# 15:           grown and released       7
# 16:                  malaisetrap       6
# 17:                    Gathering       1

dt_occ_media <- dt_occ_media[samplingProtocol %in% c("", "seen")]


# ~ collectionID ----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter. Also has too many NAs (doesn't bring useful info).
dt_occ_media[, .N, by = collectionID]
#                                      collectionID      N
# 1:                                                1093819
# 2:                                             50       1
# 3: https://bison.usgs.gov/ipt/resource?r=bugguide   25193
# 4:           e7c51ab1-870b-4ee8-9d62-092875ffa870       4

# ~ repatriated -----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter
dt_occ_media[,.N, by = repatriated][order(-N)]
# 1:       FALSE 728878
# 2:        TRUE 380577
# 3:          NA   9562
dt_occ_media[repatriated == TRUE]


# ~ datasetName -----------------------------------------------------------

# Also, this doesn't help much to identify insects from collections at this point.

dt_occ_media[,.N, by = datasetName][order(-N)]
#                                                                                                                  datasetName      N
# 1:                                                                                   iNaturalist Research-grade Observations 940921
# 2:                                                                        Observation.org, Nature data from around the World 106232
# 3:          BugGuide - Identification, Images, & Information For Insects, Spiders & Their Kin For the United States & Canada  25193
# 4:                                                                                     Norwegian Species Observation Service  20298
# 5:                                                                             ECatSym: Electronic World Catalog of Symphyta   7663
# 6:                                                                                               Earth Guardians Weekly Feed   6167
# 7:                                                                                                                     PaDIL   5607
# 8:                                                                                                       Canberra Nature Map   1852
# 9:                                                                                             Estonian Naturalists’ Society   1596
# 10:                                                               Species recordings from the Danish National portal Arter.dk   1222
# 11:                                                                                                              ClimateWatch    749
# 12:                                                                              Biodiversity4all Research-Grade Observations    645
# 13:                                                                                         ALA species sightings and OzAtlas    468
# 14:                                                                                   Atlas of Life in the Coastal Wilderness    135
# 15:                                                                                                               NatureShare    130
# 16:                                                                       India Biodiversity Portal publication grade dataset     71
# 17: Images and observations of mostly edible plants in Stephen Barstow’s Edible Garden in Norway, taken between 2005 and 2014     39
# 18:                                                                                             IBF Monitoring of Plant Galls      7
# 19:                                                                                                               Auswildlife      5
# 20:                                                                                              Australian Museum Factsheets      4
# 21:                                                    University of California Santa Barbara Invertebrate Zoology Collection      4
# 22:                                                                                             Michael Barkla's Observations      2
# 23:                                                                              Collections of Bioclass, school #179, Moscow      2
# 24:                                                               My naturesounds - nature observations with sound recordings      1
# 25:                                                           Images of Flora and Fauna of 290 Leppitt Rd, Upper Beaconsfield      1
# 26:                                                                                          UAM Insect Observations (Arctos)      1
# 27:                                                                                                                Gaia Guide      1
# 28:                                                                                           Bhutan Biodiversity Portal data      1
#                                                                                                                   datasetName      N

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
# It seems that 6390 iNaturalist records have a duplicate.
dt_occ_media[duplicated(identifier)]
dt_occ_media <- dt_occ_media[! duplicated(identifier)]

length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE now

# ~ nomenclaturalStatus ---------------------------------------------------

# nomenclaturalStatus doesn't help to filter - all NA
dt_occ_media[, .N, by = nomenclaturalStatus][order(-N)]
#    nomenclaturalStatus      N
# 1:                  NA 1112295


# ~ identifiedBy ----------------------------------------------------------

# A lot fo them are unknown
dt_occ_media[, .N, by = identifiedBy][order(-N)]


# Media license -----------------------------------------------------------

# Check licenses
dt_lic <- dt_occ_media[, .N, by = license][order(-N)]
dt_lic
#                                              license      N
# 1:    http://creativecommons.org/licenses/by-nc/4.0/ 821300
# 2:       http://creativecommons.org/licenses/by/4.0/  84533
# 3: http://creativecommons.org/licenses/by-nc-nd/4.0/  72128
# 4:                                                    42153
# 5: http://creativecommons.org/publicdomain/zero/1.0/  25621
# 6:                               All rights reserved  25021
# 7:                             © All rights reserved  17386
# 8: http://creativecommons.org/licenses/by-nc-sa/4.0/  15884
# 9:    http://creativecommons.org/licenses/by-sa/4.0/   6897
# 10:    http://creativecommons.org/licenses/by-nd/4.0/   1085
# 11:                                      CC BY-ND 4.0    273
# 12:  Photo: Bruce Thomson, http://www.auswildlife.com      5
# 13:                                 Australian Museum      4
# 14:          Creative Commons Attribution-Share Alike      2
# 15:                                                ??      2
# 16:       http://creativecommons.org/licenses/by/3.0/      1

# saveRDS(dt_lic, file = "data_processed/dt_hymenoptera_licenses.rds")

# Mark licensees about which we are sure and not sure.
license_unsure <- c("",
                    "All rights reserved", 
                    "© All rights reserved", 
                    "Photo: Bruce Thomson, http://www.auswildlife.com",
                    "Australian Museum", 
                    "??")
dt_occ_media[, license_ok := ! license %in% license_unsure]
dt_occ_media[license_ok == TRUE, .N, by = license][order(-N)] 
dt_occ_media[license_ok == FALSE, .N, by = license][order(-N)] 

# Location filters --------------------------------------------------------

# A lot of them do not have continent info!
dt_occ_media[, .N, by = continent][order(-N)]
#        continent      N
# 1:               1005422
# 2:        EUROPE  106217
# 3:          ASIA     353
# 4:        AFRICA     118
# 5: NORTH_AMERICA     108
# 6: SOUTH_AMERICA      72
# 7:       OCEANIA       5

# But for some we can get the country code
dt_occ_media[continent == ""][countryCode == "", .N] # 8277
dt_occ_media[continent == ""][level0Gid == "", .N] # 42571
dt_occ_media[continent == ""][level0Name == "", .N] # 42571
# The countryCode seems to be more comprehensive than level0Gid or level0Name
dt_occ_media[continent == ""][level0Gid == ""][countryCode == "", .N] # 8277

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
#        continent      N
# 1:       no_info 825773
# 2:        EUROPE 285867
# 3:          ASIA    352
# 4:        AFRICA    118
# 5: NORTH_AMERICA    108
# 6: SOUTH_AMERICA     72
# 7:       OCEANIA      5

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
# 1:       NA 588758
# 2:     TRUE 523537

# Also, a lot of those without continent info are in US and CA.
dt_occ_media[is.na(european)][countryCode %in% europe] # expect 0 rows
dt_occ_media[is.na(european)][! countryCode %in% europe][, .N, by = countryCode][order(-N)]
#    countryCode      N
# 1:          US 445563
# 2:          CA  80009

# Use the lists for Bombus that are marked to be in Europe in order to salvage
# URLs that can be European also.
# The info was extracted from:
# Rasmont P. & Iserbyt I. 2010-2014. Atlas of the European Bees: genus Bombus. 3d Edition. STEP
# Project, Atlas Hymenoptera, Mons, Gembloux. http://www.atlashymenoptera.net/page.aspx?ID=169
# Last accessed 2021-07-23
bombus <- fread("./data/gbif_cleaning/taxonomy/bombus_range_hiwis.csv", encoding = "Latin-1")
# Encoding fixes things like # "brodmanni\xa0"
bombus[, .N, by = rare]
bombus[, .N, by = continent]
bombus_europe <- bombus[continent == "Europe"]
bombus_europe[, .N, by = rare]
# Bombus handlirschianus doesn't seem to be in Europe
bombus_europe <- bombus_europe[rare %in% c("no", "unclear")]
bombus_europe <- bombus_europe[species != "handlirschianus"]
bombus_europe[, sp := paste(genus, species)]

# If 0 rows, then all were marked already as being also in Europe.
dt_occ_media[is.na(european)][genus_epithet %in% bombus_europe$sp]
bombus_no_europe <- dt_occ_media[is.na(european)][genus == "Bombus", sort(unique(genus_epithet))]
intersect(bombus_europe$sp, bombus_no_europe) # character(0)

# Save file ---------------------------------------------------------------

# Save only a subset of columns for the curated url-s
dt_eu_clean_url <- dt_occ_media[european == TRUE, 
                                .(media_id, gbifID, order, family, genus, specificEpithet, 
                                  lifeStage, sex, countryCode,
                                  identifier, license_ok)]

saveRDS(object = dt_eu_clean_url, 
        file = "./data/gbif_cleaning/data_processed/dt_hymenoptera_eu_clean_url.rds")
