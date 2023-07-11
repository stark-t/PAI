# This R script is designed to process and cleanse the GBIF metadata tables
# containing URLs for Lepidoptera. The result is a refined table of metadata and
# URLs, suitable for sampling and downloading images. These images were
# subsequently manually curated and annotated for inclusion in our paper.

library(bit64)
library(data.table)
library(stringr)
library(parallel)

# Read data ---------------------------------------------------------------

# Read GBIF tables. 
# Download link: https://www.gbif.org/occurrence/download/0002793-210914110416597
# See also https://github.com/stark-t/PAI/tree/main/data#readme
# Unzip the folder in ./data/gbif/lepidoptera/

# Change nThread to your needs
n_cpu <- detectCores() - 1
dt_occ_all <- fread("./data/gbif_cleaning/gbif/lepidoptera/0002793-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all <- fread("./data/gbif_cleaning/gbif/lepidoptera/0002793-210914110416597/multimedia.txt", nThread = n_cpu)
dt_media_all[, media_id := 1:.N]

# gbifID is primary key in occurrence.txt but not in multimedia.txt. 
length(unique(dt_occ_all$gbifID)) == nrow(dt_occ_all) # expect TRUE
length(unique(dt_media_all$gbifID)) == nrow(dt_media_all) # FALSE
# Note also that identifier is not a unique key in multimedia.txt.
length(unique(dt_media_all$identifier)) == nrow(dt_media_all) # FALSE
# That is why I created media_id, so that there is a primary key there as well.
 
# Read target taxa - flower visitors and pollinators
# For internal ref - see Dropbox/insect-detection/taxonomy/tk-lists-notes/lepidoptera.xlsx"
target_taxa_all <- read.csv(file = "./data/gbif_cleaning/taxonomy/lepidoptera.csv",
                            na.strings = c("NA", ""),
                            colClasses = "character")
setDT(target_taxa_all)
target_taxa <- target_taxa_all[, .(superfamily, family)]
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
# 1:           2243888
# 2:     Adult 1458163
# 3:     Imago 1002744
# 4:   Unknown  116983

# Discard unidentified genera and species
dt_occ[genus == "", .N] # 63172
dt_occ[is.na(genus), .N] # 0
dt_occ <- dt_occ[genus != ""]

dt_occ[specificEpithet == "", .N] # 185161
dt_occ[is.na(specificEpithet), .N] # 0
dt_occ <- dt_occ[specificEpithet != ""]

dt_occ[species == "", .N] # 67
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

# Keep only HUMAN_OBSERVATION - onlready done at download time fron GBIF
dt_occ_media[,.N, by = basisOfRecord][order(-N)]
#          basisOfRecord       N
# 1:   HUMAN_OBSERVATION 5998379


# ~ type ------------------------------------------------------------------

# Keep only StillImage
dt_occ_media[, .N, by = type]
# 1:  StillImage 5998321
# 2:                  54
# 3:       Sound       3
# 4: MovingImage       1
dt_occ_media <- dt_occ_media[type == "StillImage"]


# ~ file format -----------------------------------------------------------

# Keep only image/jpeg
dt_occ_media[, .N, by = format][order(-N)]
#                                format       N
# 1:                         image/jpeg 5982827
# 2:                          image/png   14528
# 3:                         image/scan     399
# 4:                          image/gif     225
# 5:                        image/pjpeg     173
# 6:       tiff, jpeg, jpeg, jpeg, jpeg     167
# 7: tiff, jpeg, jpeg, jpeg, jpeg, jpeg       1
# 8:                  image/x-canon-cr2       1
dt_occ_media[format == "image/png", .(identifier)]
dt_occ_media <- dt_occ_media[format %in% c("image/jpeg", "image/png")]

# Note that if you try to look if the .jpg or .jpeg keywords, sometimes they are
# missing, but the urls are valid (see the case for Diptera).


# ~ samplingProtocol ------------------------------------------------------

# samplingProtocol brings some useful info. Keep only "" and "seen".
dt_occ_media[, .N, by = samplingProtocol][order(-N)]
#                samplingProtocol       N
# 1:                              5420750
# 2:        by Robinson Moth Trap  249911
# 3:                         seen  161867
# 4:                    lighttrap   80521
# 5:                     on sheet   57259
# 6:                     at sugar   18992
# 7:                       caught    2717
# 8:                      indoors    2584
# 9:  caught by hand and released    1443
# 10: flashlight night observation     485
# 11:                   other trap     447
# 12:           grown and released     129
# 13:      microscopic examination      76
# 14:                  malaisetrap      31
# 15:                    collected      28
# 16: caught by hand and collected      24
# 17:                        grown      22
# 18:             (e)dna barcoding      13
# 19:       specimen in collection      12
# 20:                   pheromones      12
# 21:                       tracks       9
# 22:                   colourtrap       8
# 23:                  porch light       7
# 24:            Field Observation       3
# 25:                    at lights       2
# 26:                         hand       1
# 27:                        light       1
# 28:                     uv light       1

test <- dt_occ_media[samplingProtocol == "by Robinson Moth Trap"]
test[, .N, by = identifier]
#                                           identifier      N
# 1: https://www.nmr-pics.nl/PICTURES/voornesduin2.jpg 126180
# 2:  https://www.nmr-pics.nl/PICTURES/voornesduin.jpg 123731

# These are just two photos repeated each over 100k times ...

# The lighttrap ones might be eligible, but they do not look like being on flowers
test <- dt_occ_media[samplingProtocol == "lighttrap"]
test[, .N, by = identifier]

dt_occ_media <- dt_occ_media[samplingProtocol %in% c("", "seen")]


# ~ collectionID ----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter. Also has too many NAs (doesn't bring useful info).
dt_occ_media[, .N, by = collectionID]
#                                      collectionID      N
# 1:                                                5403676
# 2:                                             50      44
# 3: https://bison.usgs.gov/ipt/resource?r=bugguide  160995
# 4:              Vermont Atlas of Life iNaturalist    1244
# 5:           74e4045f-cd10-49fa-8533-d0fc7ec256eb   16658

# ~ repatriated -----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter
dt_occ_media[,.N, by = repatriated][order(-N)]
# 1:       FALSE 3092008
# 2:        TRUE 2482300
# 3:          NA    8309
dt_occ_media[repatriated == TRUE]


# ~ datasetName -----------------------------------------------------------

# Also, this doesn't help much to identify insects from collections at this point.

dt_occ_media[,.N, by = datasetName][order(-N)]
#                                                                                                                  datasetName      N
# 1:                                                                                   iNaturalist Research-grade Observations 4040133
# 2:                                                                        Observation.org, Nature data from around the World  922743
# 3:                                                                                     Norwegian Species Observation Service  195290
# 4:                                                                 Taiwan Moth Occurrence Data Collected From Social Network  164542
# 5:          BugGuide - Identification, Images, & Information For Insects, Spiders & Their Kin For the United States & Canada  160995
# 6:                                                                                               Earth Guardians Weekly Feed   30408
# 7:                                                                           Natural History Museum Rotterdam - Observations   16658
# 8:                                                                                                       Canberra Nature Map   12840
# 9:                                                                       India Biodiversity Portal publication grade dataset   12021
# 10:                                                               Species recordings from the Danish National portal Arter.dk    6881
# 11:                                                                              Biodiversity4all Research-Grade Observations    4612
# 12:                                                                                         ALA species sightings and OzAtlas    4444
# 13:                                                                                             Estonian Naturalists’ Society    2485
# 14:                                                                                   Atlas of Life in the Coastal Wilderness    2177
# 15:                                                                                                               NatureShare    1678
# 16:                                                                      Records of Hawk Moths (Sphingidae) from Vermont, USA    1244
# 17:        Occurrences of Threatened Species included in the Third Edition of the Red Data Book of the Komi Republic (Russia)    1137
# 18:                                                                       Australian National Insect Collection Image Library     694
# 19:                                                                                                                     PaDIL     544
# 20: Images and observations of mostly edible plants in Stephen Barstow’s Edible Garden in Norway, taken between 2005 and 2014     371
# 21:                                                                                                              ClimateWatch     243
# 22:                                                                                           Bhutan Biodiversity Portal data     167
# 23:                                                           Images of Flora and Fauna of 290 Leppitt Rd, Upper Beaconsfield      89
# 24:                            Butterfly observation records based on photographs taken by butterfly enthusiasts in Indonesia      61
# 25:                                                                                          UAM Insect Observations (Arctos)      44
# 26:                                                                                                               Auswildlife      27
# 27:                                                                                                                Gaia Guide      20
# 28:                                                                                             Michael Barkla's Observations      17
# 29:                                                                                                  ACT Bioblitz Moth Survey      16
# 30:                                                                                              Australian Museum Factsheets      10
# 31:                                                                                                                 Bioimages       8
# 32:                                                                              Collections of Bioclass, school #179, Moscow       8
# 33:                                                                                             IBF Monitoring of Plant Galls       4
# 34:                                                                                CBCGDF CCAfa Volunteer Observation Archive       4
# 35:                                                                                            Lizard Island Research Station       1
# 36:                                                                                  City of Kalamunda Biodiversity Inventory       1

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
# It seems that 6844 iNaturalist records have a duplicate.
dt_occ_media[duplicated(identifier)]
dt_occ_media <- dt_occ_media[! duplicated(identifier)]

length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE now

# ~ nomenclaturalStatus ---------------------------------------------------

# nomenclaturalStatus doesn't help to filter - all NA
dt_occ_media[, .N, by = nomenclaturalStatus][order(-N)]
#    nomenclaturalStatus      N
# 1:                  NA 5556052


# ~ identifiedBy ----------------------------------------------------------

# A lot fo them are unknown
dt_occ_media[, .N, by = identifiedBy][order(-N)]


# Media license -----------------------------------------------------------

# Check licenses
dt_lic <- dt_occ_media[, .N, by = license][order(-N)]
dt_lic
#                                                       license      N
# 1:            http://creativecommons.org/licenses/by-nc/4.0/ 3527824
# 2:         http://creativecommons.org/licenses/by-nc-nd/4.0/  631597
# 3:               http://creativecommons.org/licenses/by/4.0/  426647
# 4:                                                            393227
# 5:                                       All rights reserved  189969
# 6:                                     © All rights reserved  180354
# 7:         http://creativecommons.org/licenses/by-nc-sa/4.0/   78559
# 8:         http://creativecommons.org/publicdomain/zero/1.0/   73869
# 9:            http://creativecommons.org/licenses/by-sa/4.0/   41180
# 10:            http://creativecommons.org/licenses/by-nd/4.0/    7426
# 11:                                              CC BY-ND 4.0    4870
# 12: Creative Commons Attribution Non-Commercial Australia 3.0     447
# 13:          Photo: Bruce Thomson, http://www.auswildlife.com      27
# 14:               http://creativecommons.org/licenses/by/3.0/      17
# 15:                                                        ??      16
# 16:         http://creativecommons.org/licenses/by-nc-sa/3.0/      11
# 17:                                         Australian Museum      10
# 18:                  Creative Commons Attribution-Share Alike       2

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
# 1:               4628944
# 2:        EUROPE  914764
# 3:          ASIA    4911
# 4:        AFRICA    4338
# 5: NORTH_AMERICA    1920
# 6: SOUTH_AMERICA    1164
# 7:       OCEANIA      11

# But for some we can get the country code
dt_occ_media[continent == ""][countryCode == "", .N] # 8308
dt_occ_media[continent == ""][level0Gid == "", .N] # 194582
dt_occ_media[continent == ""][level0Name == "", .N] # 194582
# The countryCode seems to be more comprehensive than level0Gid or level0Name
dt_occ_media[continent == ""][level0Gid == ""][countryCode == "", .N] # 8308

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
# 1:       no_info 3522793
# 2:        EUROPE 2021057
# 3:          ASIA    4769
# 4:        AFRICA    4338
# 5: NORTH_AMERICA    1920
# 6: SOUTH_AMERICA    1164
# 7:       OCEANIA      11

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
# 1:       NA 2954686
# 2:     TRUE 2601366

# Also, a lot of those without continent info are in US and CA.
dt_occ_media[is.na(european)][countryCode %in% europe] # expect 0 rows
dt_occ_media[is.na(european)][! countryCode %in% europe][, .N, by = countryCode][order(-N)]
#    countryCode      N
# 1:          US 1564423
# 2:          CA  340799
# 3:          AU  213521
# 4:          TW  191053
# 5:          MX  118565

# Save file ---------------------------------------------------------------

# Save only a subset of columns for the curated url-s
dt_eu_clean_url <- dt_occ_media[european == TRUE, 
                                .(media_id, gbifID, order, family, genus, specificEpithet, 
                                  lifeStage, sex, countryCode,
                                  identifier, license_ok)]

saveRDS(object = dt_eu_clean_url, 
        file = "./data/gbif_cleaning/data_processed/dt_lepidoptera_eu_clean_url.rds")
