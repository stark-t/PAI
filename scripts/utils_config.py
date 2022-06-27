# set verbose level for print and plots
# 0=quite, 1=important, 2=all data, 3=all data and plots
verbose = 3

# seed
seed = 99

# path to original data
data_path = r'F:\202105_PAI\data\P1_Data'

# path to new sampled dataset
data_path_sampled = r'F:\202105_PAI\data\P1_Data_sampled'

# ratio to split training testing and validation
traintestval_ratio = [.6, .2, .2]

# select if images without labels should come into training, testing, validation dataset
# if True more images than labels, if False images_count = label_count
no_labels = False

