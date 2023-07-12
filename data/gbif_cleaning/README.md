This folder contains the R scripts and additional information needed to process & clean the raw GBIF datasets, sample and download the image URLs. 

However, running these R scripts is optional as the final sampled images can be downloaded with the './data/download_img.py'.
Considerable effort was dedicated nevertheless to the processing of each of the raw GBIF datasets and the reader can find the scripts in the `scripts` folder.

The folder `taxonomy` contains tabular data regarding those taxa levels that were considered as flower visitors and pollinators. Some of these tables were also needed to apply a further spatial filter - e.g. only Bombus or Syrphidae species in Europe.

The folder `data_processed` contains the sampled GBIF occurrence information together with the image URLs. These are the raw images that we first downloaded and then manually annotated and curated. The final image URLs that can be downloaded and used for training and testing the models are stored in `./data/data_tables` (img_url.txt and syrphid_img_url.txt; see `./data/README.md` for more details).

