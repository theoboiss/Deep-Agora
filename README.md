Deep-Agora
==========

*Supervised by Prof. Dr. Jean-Yves RAMEL and conducted by Théo BOISSEAU - Ecole Polytechnique de l'Université de Tours.*

Overhaul of Agora from the PaRADIIT Project: Analyzing Pattern Redundancy in texts of document images using Incremental Segmentation.

PaRADIIT is a project initiated and sponsored by 2 successive Google DH awards. It aims to turn ancient books, especially from the Renaissance, into accessible digital libraries.

The collaboration with the CESR resulted in the Agora software which simultaneously performs page layout analysis, text/graphics separation and pattern extraction.

The objective of this project is to start an overhaul of the Agora software with a new approach oriented towards deep learning.


Project structure
-----------------

    deep_learning_dev/              # directory for development of deep learning project
    ├── deep_learning_lab/          # package for deep learning lab
    │   ├── data_preparation/       # package for data preparation
    │   │   ├── __init__.py         # file to indicate this directory can be used as a package
    │   │   ├── orchestration.py    # module for coordinating the data preparation process
    │   │   ├── patch.py            # module for applying patches to images
    │   │   └── xml_parser.py       # module for parsing xml files
    │   ├── __init__.py             # file to indicate this directory can be used as a package
    │   ├── gpu_setup.py            # module for setting up GPU
    │   ├── logging.py              # module for logging information
    │   └── model.py                # module for defining deep learning model
    ├── download_data.sh            # script to download data
    └── image_segmentation.ipynb    # Jupyter notebook for image segmentation
    .gitignore                      # specifies files to ignore when committing to git
    README.md                       # readme file for the project
    environment.yml                 # conda environment file to create a conda environment
    setupjy.py                      # setup file for the project

`deep_learning_dev` is used to develop deep-learning models that will later be used for the Deep-Agora software.

It includes an API called `deep_learning_lab` that allows a data scientist to prepare data, train deep neural networks and use them.

`image_segmentation.ipynb` can be used as an exemple or an application to use the API.


Deep Learning Dev.
==================

Requirements
------------
Your machine must have a GPU and CUDA installed. We recommend using a server to work in the *deep_learning_dev* directory as the processing time can be very long (at least hours).

Check if you have a GPU and CUDA installed via the NVIDIA System Management Interface (NVIDIA-SMI) driver by entering in your terminal:

    nvidia-smi

You must also have Conda installed in order to perform the following installation.


Installation
------------
The _**`deep_learning_lab`**_ package uses the framework [dhSegment-torch](https://github.com/dhlab-epfl/dhSegment-torch), please follow the same installation guide:
>dhSegment will not work properly if the dependencies are not respected. In particular, inaccurate dependencies may result in an inability to converge, even if no error is displayed. Therefore, we highly recommend to create a dedicated environment as following :
>
>```
>conda env create --name dhs --file environment.yml
>source activate dhs
>python setup.py install
>```
Note that the environment.yml file might have been modified for compatibility with our GPU server.


### Data preparation
Raw datasets can be placed in a `raw_datasets` folder, located in the same place as the executable.
They usually contain images and XML files containing their annotations.
These annotation files cannot be used directly to train the model because they must be converted to masks.

For this reason, the _**`deep_learning_lab.data_preparation`**_ package allows the developer to select and use labels from their raw datasets to build masks.
A dataset to be patched must be specified in the API by its main directory, its image directory and its annotation directory.
It can either be added to the defaults datasets of the **`deep_learning_lab.orchestration`** module or be added via the *`Orchestrator.ingestDatasets`* method.

By default, the patched dataset is in the *results* folder and in the sub-folder named after the *"specified labels"*, under the name *training_data*.
For example: if you patched a dataset with TextLine label, the dataset will be at the location *results/TextLine/training_data/*.


### Training
Before any use of the trainer provided by the dhSegment library, you must ensure that you are correctly using your GPU.
To do this, the `deep_learning_lab.gpu_setup` module provides the user with the *`cudaDeviceSelection`* method which allows them to select or pre-select the GPU they wish to use.

The `Trainer` class from the **`deepl_learning_lab.model`** module can then be instanciated with a specified set of labels to segment and the dataset to use.
The trainer can be configured with many parameters relating to the split of the validation and test sets or to the training of the model itself.
Note that as few labels as possible should be specified at a time so that a model can be developed for each of them.
This allows greater modularity for future Deep-Agora software.

By default, the dataset to be used is *training_data* (*results/"specified labels"/training_data*). The *model* and *tensorboard* directories should be in the same location. *model* contains the best serialized models and *tensorboard* contains the logs of the metrics acquired during the training.


### Inference
The `Predictor` class from the **`deepl_learning_lab.model`** module can be instanciated with a specified set of labels to segment.

By default, the input data is *inference_data* and the output is the directory *predictions*. The output directory contains the vignettes extracted from the images.

Note that the post-processing of the inference results only supports one label for the moment.


### Data
`download_data.sh` is just an example of how to download data for the project.
However, it cannot include all the necessary datasets for good performances of the model since many datasets cannot be downloaded as easily.

Most of the datasets used are from [3]

Some sources of datasets to patch are:
- [FCR](https://zenodo.org/record/3945088) (sub500)
- [FCR](https://zenodo.org/record/4767732) (sub600)
- [ESPOSALLES](https://rrc.cvc.uab.es/?ch%3D10%26com%3Dintroduction)
- [ICFHR19 RASM2019](https://bl.iro.bl.uk/concern/datasets/f866aefa-b025-4675-b37d-44647649ba71?locale%3Den)
- [ScriptNet](https://zenodo.org/record/257972/)
- [REID2019](https://www.primaresearch.org/datasets/REID2019)
- [REID2017](https://www.primaresearch.org/datasets/REID2017)
- [Pinkas](https://zenodo.org/record/3569694)
- [ImageCLEF 2016](https://zenodo.org/record/52994)

Some dataset that are already pixel-labeled (with arbitrary label and color):
- [HBA](https://api.bnf.fr/hba-un-jeu-dimages-annotees-pour-lanalyse-de-la-structure-de-mise-en-page-douvrages-anciens)
- [SynDoc](https://drive.google.com/file/d/1_goCKP5VeStjdDS0nGeZBPqPoLCMNyb6/view) (text lines, red)
- [IllusHisDoc](https://www.dropbox.com/s/bbpb9lzanjtj9f9/illuhisdoc.zip?dl%3D0) (illustrations, red)
- [DIVA-HisDB](https://diuf.unifr.ch/main/hisdoc/diva-hisdb.html) (various labels, red)

Some datasets that are accessible via IIIF (not supported yet):
- [HORAE](https://github.com/oriflamms/HORAE/)

Pixel-labeled datasets have not yet been used because they require manual intervention: the colours and the labels must be the same as those patched.


References
==========
1. [Deep-Agora_DOC](https://github.com/theo-boi/Deep-Agora_DOC.) for details about the specifications of the project.
2. [dhSegment paper](https://arxiv.org/abs/1804.10371):
    >S. Ares Oliveira, B.Seguin, and F. Kaplan, “dhSegment: A generic deep-learning approach for document segmentation”, in Frontiers in Handwriting Recognition (ICFHR), 2018 16th International Conference on, pp. 7-12, IEEE, 2018.
3. [Datasets](https://arxiv.org/abs/2203.08504):
    >K. Nikolaidou, M. Seuret, H. Mokayed and M. Liwicki, “A survey of historical document image datasets”, IJDAR 25, 305–338 (2022).
