# This R script is designed to process and cleanse the GBIF metadata tables
# containing URLs for Coleoptera. The result is a refined table of metadata and
# URLs, suitable for sampling and downloading images. These images were
# subsequently manually curated and annotated for inclusion in our paper.

library(bit64)
library(data.table)
library(stringr)
library(parallel)

# Read data ---------------------------------------------------------------

# Read GBIF tables. 
# Download link: https://www.gbif.org/occurrence/download/0201511-210914110416597
# See also https://github.com/stark-t/PAI/tree/main/data#readme
# Unzip the folder in ./data/gbif/coleoptera/

# Change nThread to your needs
n_cpu <- detectCores() - 1
dt_occ_all <- fread("./data/gbif_cleaning/gbif/coleoptera/0201511-210914110416597/occurrence.txt", nThread = n_cpu)
dt_media_all <- fread("./data/gbif_cleaning/gbif/coleoptera/0201511-210914110416597/multimedia.txt", nThread = n_cpu)
dt_media_all[, media_id := 1:.N]

# gbifID is primary key in occurrence.txt but not in multimedia.txt. 
length(unique(dt_occ_all$gbifID)) == nrow(dt_occ_all) # expect TRUE
length(unique(dt_media_all$gbifID)) == nrow(dt_media_all) # FALSE
# Note also that identifier is not a unique key in multimedia.txt.
length(unique(dt_media_all$identifier)) == nrow(dt_media_all) # FALSE
# That is why I created media_id, so that there is a primary key there as well.

# In a first step we used all the available taxa in order to build a clean
# dataset for Europe, which we can use to define target taxa.

# Subset & taxa, lifestage filters ---------------------------------------------

dt_occ <- dt_occ_all[, .(gbifID, order, family, genus, specificEpithet, species, 
                         lifeStage, sex,
                         continent, countryCode, level0Name, level0Gid,
                         publisher, collectionID, datasetName, basisOfRecord, year, samplingProtocol,
                         identifiedBy, nomenclaturalStatus, repatriated)]
# Only continent, countryCode were used for location filtering; level0Name,
# level0Gid have more NA-s than countryCode.

rm(dt_occ_all); gc()

# Discard unwanted life stages. Keep the unknowns also.
dt_occ[, .N, by = lifeStage][order(-N)]
dt_occ <- dt_occ[lifeStage %in% c("Imago", "Adult", "Unknown" , "")]
dt_occ[, .N, by = lifeStage][order(-N)]
#    lifeStage      N
# 1:           878921
# 2:     Imago 398610
# 3:     Adult 312848
# 4:   Unknown  28049

# Discard unidentified genera and species
dt_occ[genus == "", .N] # 10729
dt_occ[is.na(genus), .N] # 0
dt_occ <- dt_occ[genus != ""]

dt_occ[specificEpithet == "", .N] # 59094
dt_occ[is.na(specificEpithet), .N] # 0
dt_occ <- dt_occ[specificEpithet != ""]

dt_occ[species == "", .N] # 0
# dt_occ[species == ""]
# dt_occ <- dt_occ[species != ""]

dt_occ[family == "", .N] # 0

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

# Keep only HUMAN_OBSERVATION - already done at download time from GBIF
dt_occ_media[,.N, by = basisOfRecord][order(-N)]
#          basisOfRecord       N
# 1:   HUMAN_OBSERVATION 2419681


# ~ type ------------------------------------------------------------------

# Keep only StillImage
dt_occ_media[, .N, by = type]
# 1:  StillImage 2419566
# 2:                  76
# 3:       Sound      37
# 4: MovingImage       2
dt_occ_media <- dt_occ_media[type == "StillImage"]


# ~ file format -----------------------------------------------------------

# Keep only image/jpeg, "image/png"
dt_occ_media[, .N, by = format][order(-N)]
#               format       N
# 1:        image/jpeg 2411571
# 2:         image/png    7536
# 3:         image/gif     263
# 4:        image/scan     126
# 5:       image/pjpeg      67
# 6: image/x-raw-canon       3
dt_occ_media[format == "image/png", .(identifier)]
dt_occ_media <- dt_occ_media[format %in% c("image/jpeg", "image/png")]

# Note that if you try to look if the .jpg or .jpeg keywords, sometimes they are
# missing, but the urls are valid (see the case for Diptera).


# ~ samplingProtocol ------------------------------------------------------

# samplingProtocol brings some useful info. Keep only "" and "seen".
dt_occ_media[, .N, by = samplingProtocol][order(-N)]
#                                                                                                       samplingProtocol       N
# 1:                                                                                                                     2173590
# 2:                                                                                                                seen  137617
# 3:                                                                                                            observed   47341
# 4:                                                                                         caught by hand and released    6725
# 5:                                                                                                     lamp/light trap    5798
# 6:                                                                                             microscopic examination    5338
# 7:                                                                                                           lighttrap    5198
# 8:                                                                                                           sweep-net    5078
# 9:                                                                                               beating of vegetation    5008
# 10:                                                                                                       sweep netting    4366
# 11:                                                                                                        litter sieve    3547
# 12:                                                                                                      beating screen    2682
# 13:                                                                                        caught by hand and collected    2346
# 14:                                                                                                         net by hand    2342
# 15:                                                                                                            on sheet    2012
# 16:                                                                                                        pitfall trap    1743
# 17:                                                                                                               trawl    1602
# 18:                                                                                                             indoors    1288
# 19:                                                                                                          light trap    1252
# 20:                                                                                                            pit trap     542
# 21:                                                                                                            at sugar     534
# 22:                                                                                              specimen in collection     461
# 23:                                                                                              night search with lamp     459
# 24:                                                                                                                trap     334
# 25:                                                                                                             pondnet     320
# 26:                                                                                                          other trap     275
# 27:                                                                                                            flooding     232
# 28:                                                                                                                lure     189
# 29:                                                                                                         window trap     169
# 30:                                                                                                               heard     115
# 31:                                                                                     collected for breeding/hatching      94
# 32:                                                                                                          colourtrap      76
# 33:                                                                                                         malaisetrap      74
# 34:                                                                                                          pheromones      55
# 35:                                                                                                            tramping      50
# 36:                                                                                                        malaise trap      40
# 37:                                                                                                    tracks and signs      36
# 38:                                                                                             grown and in collection      33
# 39:                                                                                                       coloured bowl      29
# 40:                                                                                                           in pellet      16
# 41:                                                                                            observation in the night      15
# 42:                                                                                                  grown and released      15
# 43:                                                                                                          extraction      13
# 44:                                                                                                    pooter/aspirater      12
# 45:                                                                                                 vehicle mounted net      10
# 46:                                                                                                    (e)dna barcoding       8
# 47:                                                           handnet; photos taken in container with scale/color chart       7
# 48:                                                                                                      surber sampler       4
# 49: handnet; photos taken on hand using scale/color chart finger ring; photos taken in container with scale/color chart       4
# 50:                                                                                                      moss squeezing       3
# 51:                                                                                          observation in the daytime       3
# 52:                                                                                                            fyke net       2
# 53:                                                                                                                hand       1
# 54:                                                                                                  bottom grab sample       1
# 55:                                                                  observation in the nighttime; photos taken on hand       1
# 56:                                                                                            observation at nightfall       1
# 57:                                                                                                             handnet       1

# lamp/light trap, beating of vegetation, etc for example might not be pollinators anyways
dt_occ_media <- dt_occ_media[samplingProtocol %in% c("", "observed", "seen", "caught by hand and released",
                                                     "sweep-net", "sweep netting", "net by hand")]


# ~ collectionID ----------------------------------------------------------

# This is not a good filter; it has too many NAs (doesn't bring useful info).
dt_occ_media[, .N, by = collectionID]
#                                      collectionID      N
# 1:                                                2304646
# 2:                                             50       9
# 3:  urn:uuid:18e3cd08-a962-4f0a-b72c-9a0b3600c5ad     272
# 4: https://bison.usgs.gov/ipt/resource?r=bugguide   72132

# ~ repatriated -----------------------------------------------------------

# At a random check, some images are of insects on flowers, so this is not a
# good filter
dt_occ_media[,.N, by = repatriated][order(-N)]
# 1:       FALSE 1556446
# 2:        TRUE  812650
# 3:          NA    7963
dt_occ_media[repatriated == TRUE]


# ~ datasetName -----------------------------------------------------------

# Also, this doesn't help much to identify insects from collections at this point.

dt_occ_media[,.N, by = datasetName][order(-N)]
#                                                                                                                  datasetName       N
# 1:                                                                                   iNaturalist Research-grade Observations 1689166
# 2:                                                                        Observation.org, Nature data from around the World  409071
# 3:                                                                          Artportalen (Swedish Species Observation System)  138333
# 4:          BugGuide - Identification, Images, & Information For Insects, Spiders & Their Kin For the United States & Canada   72132
# 5:                                                                                     Norwegian Species Observation Service   42058
# 6:                                                                                               Earth Guardians Weekly Feed   11778
# 7:                                                               Species recordings from the Danish National portal Arter.dk    4995
# 8:                                                                                                       Canberra Nature Map    3252
# 9:                                                                                         ALA species sightings and OzAtlas    1764
# 10:                                                                              Biodiversity4all Research-Grade Observations    1643
# 11:                                                                                             Estonian Naturalists’ Society    1001
# 12:                                                                        ConDidact Citizen Science Surveys - Ladybirds 2013     707
# 13:                                                                                   Atlas of Life in the Coastal Wilderness     513
# 14:                                                                                   NMNH Extant Specimen Records (USNM, US)     272
# 15:                                                                                                               NatureShare     217
# 16:                                                                       India Biodiversity Portal publication grade dataset      64
# 17: Images and observations of mostly edible plants in Stephen Barstow’s Edible Garden in Norway, taken between 2005 and 2014      33
# 18:                                                                   Lajitietokeskus/FinBIF - Notebook, general observations      13
# 19:                                                           Images of Flora and Fauna of 290 Leppitt Rd, Upper Beaconsfield      12
# 20:                                                                                          UAM Insect Observations (Arctos)       9
# 21:                                                                                             Southern Highlands Nature Map       7
# 22:                                                                              Collections of Bioclass, school #179, Moscow       6
# 23:                                                                                             Michael Barkla's Observations       5
# 24:                                                                                                 Albury Wodonga Nature Map       5
# 25:                                                                                             IBF Monitoring of Plant Galls       1
# 26:                                                                               IASTracker. Invasive Alien Species database       1
# 27:                                                                                                    Noosa Shire Nature Map       1
# ~ publisher -------------------------------------------------------------

dt_occ_media[,.N, by = publisher.x][order(-N)] # publisher.x is from occurrence file
dt_occ_media[,.N, by = publisher.y][order(-N)] # publisher.y is from media file

# ~ year ------------------------------------------------------------------

# Be aware of the NA-s, they are valid. The old ones are really a few though.
dt_occ_media[, .N, by = year][order(-N)]
dt_occ_media[is.na(year), .N] # 843
dt_occ_media[year <= 1990, .N] # 3703
dt_occ_media <- dt_occ_media[year >= 1990 | is.na(year)]

# ~ identifier ------------------------------------------------------------

# Check if identifier is missing
dt_occ_media[identifier == "", .N] # 0
dt_occ_media[is.na(identifier), .N] # 0

# Check if identifier is unique - not the case!
length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE, but gave FALSE
# It seems that 368 (mostly iNaturalist) records have a duplicate.
dt_occ_media[duplicated(identifier)]
dt_occ_media <- dt_occ_media[! duplicated(identifier)]

length(unique(dt_occ_media$identifier)) == nrow(dt_occ_media) # expect TRUE now

# ~ nomenclaturalStatus ---------------------------------------------------

# nomenclaturalStatus doesn't help to filter - all NA
dt_occ_media[, .N, by = nomenclaturalStatus][order(-N)]
#     nomenclaturalStatus      N
# 1:                     2235161
# 2:               valid  138060


# ~ identifiedBy ----------------------------------------------------------

# A lot fo them are unknown
dt_occ_media[, .N, by = identifiedBy][order(-N)]

# Media license -----------------------------------------------------------

# Check licenses
dt_lic <- dt_occ_media[, .N, by = license][order(-N)]
dt_lic
#                                               license      N
# 1:    http://creativecommons.org/licenses/by-nc/4.0/ 1462755
# 2: http://creativecommons.org/licenses/by-nc-nd/4.0/  333545
# 3:       http://creativecommons.org/licenses/by/4.0/  207327
# 4:                             © all rights reserved  137921
# 5:                                                     89444
# 6: http://creativecommons.org/publicdomain/zero/1.0/   45209
# 7:                             © All rights reserved   37157
# 8: http://creativecommons.org/licenses/by-nc-sa/4.0/   34431
# 9:    http://creativecommons.org/licenses/by-sa/4.0/   21945
# 10:    http://creativecommons.org/licenses/by-nd/4.0/    3071
# 11:                            Usage Conditions Apply     272
# 12:                                        Ola Ejdrén      37
# 13:                                  Jan-Åke Noresson       5
# 14:                                    Márika Bernadt       5
# 15:                                  Bertil Johansson       4
# 16:                                   Micke Johansson       4
# 17:                        Lars-Inge Larsson, Ljugarn       4
# 18:                                Sven-Erik Holmåker       4
# 19:                                   Kalle Bergström       3
# 20:                                   Lisa Steinholtz       3
# 21:                         Marianne Larsson, Ljugarn       3
# 22:                                      Jan Olausson       3
# 23:                                    Carina Lerdahl       3
# 24:                  Creative Commons Attribution 3.0       3
# 25:                                     Anne Sandberg       2
# 26:                                     Annika Brandt       2
# 27:                                       Mats Aldrin       2
# 28:                                          Ole Paus       2
# 29:                                   Toni Hermansson       2
# 30:                                   Nils-Gunnar Eek       2
# 31:                              Marianne Silfverlåås       2
# 32:                                                ??       2
# 33:                                      Thomas Kraft       2
# 34:                      Upptäckt av Kristina Brorson       2
# 35:                          Therese Kustvall Larsson       2
# 36:                                Matilda Alfredsson       2
# 37:                                 AnnaKarin Åstrand       1
# 38:                                     Rolf Kokkonen       1
# 39:                                    Björn Phragmén       1
# 40:                                Jan-Inge Tobiasson       1
# 41:                                 Torbjörn Karlsson       1
# 42:                                  Anders Andersson       1
# 43:                                      Hans Nilsson       1
# 44:                                      Thomas Strid       1
# 45:                                   Anders Forsgren       1
# 46:                                    Ingemar Alenäs       1
# 47:                                    Ulrika Widgren       1
# 48:                                   Axel Lagerquist       1
# 49:    Petra Pohjola, Länsstyrelsen i Norrbottens län       1
# 50:                               Jan-Olof Pettersson       1
# 51:                                         L.Ahlberg       1
# 52:                         Lars-Inge Larsson Ljugarn       1
# 53:                         Olof Persson (Vassmolösa)       1
# 54:                                      Olof Persson       1
# 55:                                                 b       1
# 56:                               Sandra Christensson       1
# 57:                                     göran lindell       1
# 58:                                 Mona-lis Österman       1
# 59:                                   Ulf Modin Tumba       1
# 60:                             Anders Löfgren, Tumba       1
# 61:                                        Trollkulan       1
# 62:                                  Christian Allard       1
# 63:                         Marianne Larsson  Ljugarn       1
# 64:                                    Emma Johansson       1
# 65:                                  Linnéa Andersson       1
# 66:                                Per Olsson, Flyet.       1
# 67:                                      Bo Brinkhoff       1
# 68:                                      Fredrik Paus       1
# 69:                                    Timmy Rydström       1
# 70:                                     Bosse Ohlsson       1
# 71:                                   Henrik Elofsson       1
# 72:                                   Gunnar Isacsson       1
# 73:                                       Frida Snell       1
# 74:                                        Mattias Ek       1
# 75:                                   Anders Törnberg       1

# Mark licensees about which we are sure and not sure.
license_ok <- c("http://creativecommons.org/licenses/by-nc/4.0/",
                "http://creativecommons.org/licenses/by-nc-nd/4.0/",
                "http://creativecommons.org/licenses/by/4.0/",
                "http://creativecommons.org/publicdomain/zero/1.0/",
                "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "http://creativecommons.org/licenses/by-sa/4.0/",
                "http://creativecommons.org/licenses/by-nd/4.0/")
dt_occ_media[, license_ok := license %in% license_ok]
dt_occ_media[license_ok == TRUE, .N, by = license][order(-N)] 
dt_occ_media[license_ok == FALSE, .N, by = license][order(-N)] 
dt_occ_media[, .N, by = license_ok]
#    license_ok       N
# 1:       TRUE 2108283
# 2:      FALSE  264938


# Location filters --------------------------------------------------------

# A lot of them do not have continent info!
dt_occ_media[, .N, by = continent][order(-N)]
#         continent      N
# 1:               1867503
# 2:        EUROPE  505678
# 3: NORTH_AMERICA      30
# 4:    ANTARCTICA       7
# 5:        AFRICA       2
# 6:          ASIA       1

# But for some we can get the country code
dt_occ_media[continent == ""][countryCode == "", .N] # 4184
dt_occ_media[continent == ""][level0Gid == "", .N] # 90385
dt_occ_media[continent == ""][level0Name == "", .N] # 90385
# The countryCode seems to be more comprehensive than level0Gid or level0Name
dt_occ_media[continent == ""][level0Gid == ""][countryCode == "", .N] # 4184

dt_occ_media[countryCode == "", unique(level0Gid)]
dt_occ_media[level0Gid == "", unique(countryCode)]

# Read country code data
country <- fread('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')

unique(country$region)
country[region == ""]
europe <- country[region == "Europe", "alpha-2"][[1]]

# Update EUROPE info based on country code.
dt_occ_media[countryCode %in% europe, continent := "EUROPE"]
dt_occ_media[continent == "", continent := "no_info"]
dt_occ_media[, .N, by = continent][order(-N)]
#        continent       N
# 1:       no_info 1195921
# 2:        EUROPE 1177260
# 3: NORTH_AMERICA      30
# 4:    ANTARCTICA       7
# 5:        AFRICA       2
# 6:          ASIA       1

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
# 1:     TRUE 1483634
# 2:       NA  889587

# Also, a lot of those without continent info are in US and CA.
dt_occ_media[is.na(european)][countryCode %in% europe] # expect 0 rows
dt_occ_media[is.na(european)][! countryCode %in% europe][, .N, by = countryCode][order(-N)]
#    countryCode      N
# 1:          US 548495
# 2:          CA  82704
# 3:          AU  59590
# 4:          MX  37221
# 5:          ZA  24989

dt_occ_media[is.na(european), unique(countryCode)] %in% europe
sp_non_europe <- dt_occ_media[is.na(european), unique(genus_epithet)]
length(sp_non_europe) # 16684
sp_europe <- dt_occ_media[european == TRUE, unique(genus_epithet)]
length(sp_europe) # 6326

# Examples of top frequent non-european species
non_european <- dt_occ_media[is.na(european), .N, by = genus_epithet][order(-N)]
non_european
# You can check the maps of some on GBIF. it seems that I do a good job of
# catching what is European only.

#                   genus_epithet     N
# 1:   Diabrotica undecimpunctata 19818 # native to North America, https://www.gbif.org/occurrence/map?taxon_key=1048501
# 2:         Cicindela sexguttata 15891 # common North American species, https://www.gbif.org/occurrence/map?taxon_key=1034887
# 3:     Tetraopes tetrophthalmus 13282 # northeastern part of North America, https://www.gbif.org/occurrence/map?taxon_key=5002722
# 4: Chauliognathus pensylvanicus 12341 # North America https://www.gbif.org/occurrence/map?taxon_key=7596343
# 5:     Odontotaenius disjunctus 10333 # temperate North American forests, https://www.gbif.org/occurrence/map?taxon_key=6992074


# Save file ---------------------------------------------------------------

# Save only a subset of columns for the curated url-s
dt_eu_clean_url <- dt_occ_media[european == TRUE, 
                                .(media_id, gbifID, order, family, genus, specificEpithet, 
                                  lifeStage, sex, countryCode,
                                  identifier, license_ok)]

saveRDS(object = dt_eu_clean_url, 
        file = "./data/gbif_cleaning/data_processed/dt_coleoptera_clean_url_all_euopean_taxa.rds")
