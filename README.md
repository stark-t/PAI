# PAI: Pollination Artificial Intelligence

This repository is associated with the paper:

- preprint
> Stark, T., Stefan, V., Wurm, M., Spanier, R., Taubenboeck, H., & Knight, T. M. (2023). YOLO object detection models can locate and classify broad groups of flower-visiting arthropods in images. https://assets.researchsquare.com/files/rs-2673814/v1/ed1dd10c2de6319cc445a1ac.pdf?c=1678851330

Computations were done using resources of the Leipzig University Computing Centre. Note that the setting of the environments are valid only for the resources made available to us. You will need to edit most of the data paths to fit your local settings.
# How to use this repository

Before you proceed, make sure you have Git installed on your machine - see this https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

You can clone this repository with these git command line suggestions:
```sh
cd ~/path/to/desired/folder

git clone https://github.com/stark-t/PAI
```

For training the models on a GPU cluster see our steps documented at https://github.com/stark-t/PAI/tree/main/scripts/cluster#readme

The script used for model evaluation is `./scripts/evaluate.py`. Scripts named as `untils_*.py` contain helper functions.

For using our trained weights, see the suggestions at https://github.com/stark-t/PAI/tree/main/detectors#readme

# Data

Check https://github.com/stark-t/PAI/tree/main/data#readme