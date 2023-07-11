# Script to get license stats for the GBIF URLs

library(data.table)


# All license types -------------------------------------------------------

# All other orders, except the Syprhid dataset
dt_img_url <- fread('./data/data_tables/img_url.txt')

# The syprhid dataset
dt_syrphid_img_url <- fread('./data/data_tables/syrphid_img_url.txt')

dt <- rbindlist(list(dt_img_url, dt_syrphid_img_url), fill = TRUE)

dt[, .N, keyby = license]
#                                              license     N
# 1:                                      CC BY-ND 4.0    33
# 2: http://creativecommons.org/licenses/by-nc-nd/4.0/  7200
# 3: http://creativecommons.org/licenses/by-nc-sa/4.0/   880
# 4:    http://creativecommons.org/licenses/by-nc/4.0/ 10805
# 5:    http://creativecommons.org/licenses/by-nd/4.0/     5
# 6:    http://creativecommons.org/licenses/by-sa/4.0/   585
# 7:       http://creativecommons.org/licenses/by/4.0/  1838
# 8: http://creativecommons.org/publicdomain/zero/1.0/   250

dt[, .(license = sort(unique(license)))]
#                                                license
#   1:                                      CC BY-ND 4.0
#   2: http://creativecommons.org/licenses/by-nc-nd/4.0/
#   3: http://creativecommons.org/licenses/by-nc-sa/4.0/
#   4:    http://creativecommons.org/licenses/by-nc/4.0/
#   5:    http://creativecommons.org/licenses/by-nd/4.0/
#   6:    http://creativecommons.org/licenses/by-sa/4.0/
#   7:       http://creativecommons.org/licenses/by/4.0/
#   8: http://creativecommons.org/publicdomain/zero/1.0/


# How many img from iNaturalist? ----------------------------------------------

dt[grepl(pattern = 'inaturalist', x = url), .N] # 11631
dt[grepl(pattern = 'inaturalist', x = url), .N] / nrow(dt) * 100 # 53.8572



# How many img from observation.org? --------------------------------------

dt[grepl(pattern = 'observation', x = url), .N] # 9267
dt[grepl(pattern = 'observation', x = url), .N] / nrow(dt) * 100 # 42.91072
