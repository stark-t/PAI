# Script to look at the dispersion and distribution of relative box area

library(data.table)
library(magrittr)
library(ggplot2)

# The dataset used for train-val-test (not including the syrphid extra test)
dt <- fread('./data/data_tables/img_annotation.txt', n)

# Compute box areas
dt[, relative_box_area := (width_rel * height_rel)]

dt_stats <- dt[used_for_ai == 1, .(min_rba = min(relative_box_area),
                                   q1 = quantile(relative_box_area, probs = 0.25),
                                   mean_rba = mean(relative_box_area),
                                   median_rba = median(relative_box_area),
                                   q3 = quantile(relative_box_area, probs = 0.75),
                                   max_rba = max(relative_box_area)), 
               keyby = label_name]
dt_stats
#                label_name      min_rba          q1  mean_rba median_rba        q3   max_rba
# 1:                Araneae 6.014506e-03 0.160800550 0.3815403 0.34365443 0.5650553 0.9883117
# 2:             Coleoptera 6.353022e-04 0.083189741 0.2673314 0.21026016 0.4099316 0.9788594
# 3:                Diptera 8.364677e-03 0.275333200 0.4654382 0.47047154 0.6536037 0.9833333
# 4:              Hemiptera 2.447764e-04 0.046266556 0.2923931 0.20992366 0.4720694 0.9865756
# 5:            Hymenoptera 7.387430e-03 0.203583885 0.4064247 0.38033854 0.5796340 0.9969925
# 6: Hymenoptera_Formicidae 3.404617e-04 0.007605434 0.1176675 0.02218739 0.1155895 0.9733047
# 7:            Lepidoptera 1.959483e-03 0.176525000 0.3904635 0.34129213 0.5755714 0.9937500
# 8:             Orthoptera 4.323324e-05 0.251600477 0.4544817 0.44886780 0.6398625 0.9952847

quantile(dt[used_for_ai == 1, relative_box_area], probs = c(0.25, 0.75))
#          25%        75% 
#   0.09729696 0.53594361

summary(dt[used_for_ai == 1, relative_box_area])
#       Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# 0.0000432 0.0972970 0.2884545 0.3370342 0.5359436 0.9969925 

ggplot() +
  geom_histogram(data = dt[used_for_ai == 1],
                 aes(x = relative_box_area)) +
  geom_vline(data = dt_stats,
             aes(xintercept = mean_rba),
             color = 'red') +
  facet_wrap(. ~ label_name, scales = 'free_y')
