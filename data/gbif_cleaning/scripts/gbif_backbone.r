# Script to process the downloaded BGIF taxa backbone. This should include all
# taxa names that are on GBIF. The dataset is big, so I pruned it to my needs.
# This info was needed to check species names when cleaning the GBIF occurrence
# tables.

# Link to raw backbone dataset "Backbone 26th December 2021":
# https://hosted-datasets.gbif.org/datasets/backbone/2021-11-26/

# Download the backbone.zip file (~ 883 Mb) in ./data/gbif_cleaning/gbif/

library(data.table)
library(magrittr)

dt_taxon_all <- fread("./data/gbif_cleaning/gbif/backbone/Taxon.tsv", 
                      sep = "\t",
                      header=TRUE,
                      colClasses = "character")

dt_taxon <- dt_taxon_all[order %in% c("Coleoptera", "Diptera", "Hymenoptera", "Lepidoptera")]
dt_taxon[is.na(genus), .N] # 0, but because they are ""
dt_taxon[genus == "", .N]
# dt_taxon <- dt_taxon[genus != ""] some of them carry Family info that might be useful
dt_taxon[family == "", .N]
# dt_taxon <- dt_taxon[family != ""] # some of them carry canonicalName that might be useful

dt_taxon$taxonomicStatus %>% unique() %>% sort()
# [1] "accepted" "doubtful" "heterotypic synonym" "homotypic synonym" "proparte synonym" "synonym"

dt_taxon$taxonRank %>% unique() %>% sort()
# [1] "family"     "form"       "genus"      "order"      "species"    "subspecies" "unranked"   "variety" 

# Taxonomic status exists both for genus, where taxonRank=genus and species
# (taxonRank=species), but not for family

# No need to carry info about kingdom phylum class
dt_taxon$kingdom %>% unique() # "Animalia"  
dt_taxon$phylum %>% unique() # "Arthropoda"
dt_taxon$class %>% unique() # "Insecta"
cols_exclude <- c("kingdom", "phylum", "class")
dt_taxon <- dt_taxon[, setdiff(names(dt_taxon), cols_exclude), with = FALSE]

# Save the pruned taxa backbone table as R binary format
saveRDS(dt_taxon, file = "./data/gbif_cleaning/gbif/backbone/dt_taxon.rda")
