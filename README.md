**Deep-Agora**
==============
*Supervised by Prof. Dr. Jean-Yves RAMEL and conducted by Théo BOISSEAU at Ecole Polytechnique de l'Université de Tours.*



Introduction
============
Overhaul of Agora from the PaRADIIT Project: Analyzing Pattern Redundancy in texts of document images using Incremental Segmentation.

PaRADIIT is a project initiated and sponsored by 2 successive Google DH awards.
It aims to turn ancient books, especially from the Renaissance, into accessible digital libraries.

The collaboration with the CESR resulted in the Agora software which simultaneously performs page layout analysis, text/graphics separation and pattern extraction.

The objective of this project is to start an overhaul of the Agora software with a new approach oriented towards deep learning.


Project structure
-----------------

    deep-agora
    ├── deep_learning/                  # working directory for development of the data science project
    │   ├── deep_learning_lab/          # package for deep learning lab
    │   │   ├── data_preparation/       # subpackage of deep learning lab for data preparation
    │   │   │   ├── __init__.py         # file to indicate this directory can be used as a package
    │   │   │   ├── orchestration.py    # module for coordinating the data preparation process
    │   │   │   ├── patch.py            # module for applying patches to images
    │   │   │   └── xml_parser.py       # module for parsing xml files
    │   │   ├── __init__.py             # file to indicate this directory can be used as a package
    │   │   ├── gpu_setup.py            # module for setting up GPU
    │   │   ├── logging.py              # module for logging information
    │   │   └── model.py                # module for defining deep learning model
    │   ├── raw_datasets/               # location to download raw data sets
    │   ├── tests/                      # tests of deep_learning_lab designed for Pytest
    │   │   ├── integration/            # integration tests
    │   │   └── unit/                   # unit tests
    │   ├── download_data.sh            # example of script to download data (incomplete)
    │   └── segmentation.ipynb          # Jupyter notebook for semantic segmentation
    ├── ...                             # future working directories (e.g. software development)
    ├── dependencies/                   # project dependencies
    │   ├── dhSegment-torch/            # sub-module and framework for semantic segmentation
    │   ├── environment.yml             # conda environment file adapted to sm_86 CUDA architecture
    │   └── setup.py                  # setup file adapted to sm_86 CUDA architecture
    ├── .gitignore                      # specifies files to ignore when committing to git
    ├── .gitmodules                     # specifies submodules in dependencies/
    └── README.md                       # readme file for the project

-   `deep_learning/` is a working directory of data science.
    It is designed to develop deep-learning models that will be used in the future working directory `deep_agora/` of software development.
    It includes the _**`deep_learning_lab`**_ package that allows a data scientist to prepare data, train deep neural networks and use them for inference on images.
    `image_segmentation.ipynb` can be used as an example or an application to use the package.

-   `dhSegment-torch/` is an external Deep-Learning framework cloned from the GitHub repository [dhSegment-torch](https://github.com/dhlab-epfl/dhSegment-torch).
    Its environment files have been edited to adapt to sm_86 CUDA architecture.
    >**dhSegment** is a tool for Historical Document Processing.
    >Its generic approach allows to segment regions and extract content from different type of documents.



Deep Learning working directory
===============================
Requirements
------------
You need to use a Linux or WSL machine and we highly recommend using a machine with a GPU to work in the *deep_learning* directory as the processing time can be very long (many hours).

Check if you have a GPU and CUDA installed via the [NVIDIA System Management Interface (NVIDIA-SMI) driver](https://www.nvidia.fr/Download/index.aspx) by entering in your terminal:

    nvidia-smi

You must also have Conda installed in order to perform the following installation.


Installation
------------
The _**`deep_learning_lab`**_ package uses the sub-module and framework *dhSegment-torch* [[2]](#references).

Go to `dependencies/` and clone the sub-module(s) as follows:

    cd dependencies/
    git submodule update --init --recursive

We edited its environment files (`environment.yml` and `setup.py`) for compatibility with the sm_86 CUDA architecture of our machine.
You can [match your GPU name to your CUDA architecture](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
To apply such changes, do as follows:

    cp environment.yml dhSegment-torch/
    cp setup.py dhSegment-torch/

Now, to install package of the sub-module, go to `dhSegment-torch/` as follows:
    
    cd dhSegment-torch

And follow its [installation guide](https://github.com/dhlab-epfl/dhSegment-torch#installation):
>   Installation
>   ------------
>   dhSegment will not work properly if the dependencies are not respected.
>   In particular, inaccurate dependencies may result in an inability to converge, even if no error is displayed.
>   Therefore, we highly recommend to create a dedicated environment as follows:
>
>     conda env create --name dhs --file environment.yml
>     source activate dhs
>     python setup.py install
>


Data preparation
----------------
Raw datasets can be placed in a `raw_datasets/` folder, located in the working directory.
They usually contain images and XML files containing their annotations.
These annotation files cannot be used directly to train the model because they must be converted to masks.

For this reason, the _**`deep_learning_lab.data_preparation`**_ package allows the developer to select and use labels from their raw datasets to build masks.
A dataset to be patched must be specified by its main directory, its image directory and its annotation directory.

For the moment, some default datasets are implemented inside the source code of the **`deep_learning_lab.data_preparation.orchestration`** module.
To be used, these default datasets must already have been downloaded.
Additional datasets can either be added to the defaults datasets in the source code of the module or via the *`Orchestrator.ingestDatasets`* method.

In addition, to easily analyse the contents of a raw dataset, the *`Orchestrator.ingestLabels`* method provides a `prompt` parameter that allows the user to choose their labels and the *`Orchestrator.validate`* method prints statistics on each dataset and its contents.

By default, the patched dataset is in the *results* folder and in the sub-folder named after the *"specified labels"*, under the name *training_data*.
For example: if you patched a dataset with TextLine label, the dataset will be at the location *results/TextLine/training_data/*.

Note that the **`deep_learning_lab.data_preparation.patch`** module running in the backend of the *`Orchestrator`* class has not been validated for multi-labels at the moment.


Training
--------
Before training, you must specify if you want to use a CPU, a GPU and which one.
To do this, the **`deep_learning_lab.gpu_setup`** module allows the selection of a GPU/CPU in backend when instanciating the `Trainer` class.

The `Trainer` class from the **`deepl_learning_lab.model`** module can then be instanciated with a specified set of labels to segment and the dataset to use.
The trainer can be configured with many parameters relating to the split of the validation and test sets or to the training of the model itself.
Note that as few labels as possible should be specified at a time so that a model can be developed for each of them.
This allows greater modularity for future Deep-Agora software.

By default, the dataset to be used is *training_data* (*results/"specified labels"/training_data*).
The *model* and *tensorboard* directories should be in the same location.
*model* contains the best serialized models and *tensorboard* contains the logs of the metrics acquired during the training.


Inference
---------
The `Predictor` class from the **`deepl_learning_lab.model`** module can be instanciated with a specified set of labels to segment.

By default, the input data is *inference_data* and the output directory is *predictions*.
The output directory contains the vignettes extracted from the images.
The output of the *`Predictor.start`* method returns additional data such as the original image on which the bounding boxes and polygons are drawn.

Note that the inference post-processing has only been validated for one label at the moment.


Data
----
`download_data.sh` is just an example of how to download data for the project.
It is unlikely that such a script could include all the datasets needed for good model performance, as many datasets cannot be downloaded as easily.

Most of the datasets bellow have been chosen from *A survey of historical document image datasets* [[3]](#references):

Some sources of datasets to patch are:
- [FCR](https://zenodo.org/record/3945088)
- [ICFHR19](https://rrc.cvc.uab.es/?ch%3D10%26com%3Dintroduction)
- [ScriptNet](https://zenodo.org/record/257972/)
- [REID2019](https://www.primaresearch.org/datasets/REID2019)
- [REID2017](https://www.primaresearch.org/datasets/REID2017)
- [Pinkas](https://zenodo.org/record/3569694)
- [ImageCLEF 2016](https://zenodo.org/record/52994)

And their content:
| Dataset | TextLine | TextRegion | Word | ImageRegion |
|---|---|---|---|---|
| FCR_500/data | 32177 | 1701 | - | - |
| ABP_FirstTestCollection | 961 | 226 | - | - |
| Bohisto_Bozen_SetP | 815 | 152 | - | - |
| EPFL_VTM_FirstTestCollection | 252 | 38 | - | - |
| HUB_Berlin_Humboldt | 693 | 81 | - | - |
| NAF_FirstTestCollection | 930 | 164 | - | - |
| StAM_Marburg_Grimm_SetP | 857 | 214 | - | - |
| UCL_Bentham_SetP | 1024 | 191 | - | - |
| unibas_e-Manuscripta | 848 | 96 | - | - |
| ABP_FirstTestCollection | 4230 | 30 | - | - |
| Bohisto_Bozen_SetP | 910 | 26 | - | - |
| BHIC_Akten | 2339 | 30 | - | - |
| EPFL_VTM_FirstTestCollection | 2790 | 28 | - | - |
| HUB_Berlin_Humboldt | 885 | 28 | - | - |
| NAF_FirstTestCollection | 6147 | 29 | - | - |
| StAM_Marburg_Grimm_SetP | 1064 | 30 | - | - |
| UCL_Bentham_SetP | 2294 | 31 | - | - |
| unibas_e-Manuscripta | 1081 | 20 | - | - |
| pinkas_dataset | 1013 | 175 | 13744 | - |
| IEHHR-XMLpages | 3070 | 968 | 31501 | - |
| ImageCLEF 2016 pages_train_jpg | 9645 | 765 | - | - |
| REID2019 | - | 454 | - | 3 |

Some dataset that are already pixel-labeled (with arbitrary label and color):
- [HBA](https://api.bnf.fr/hba-un-jeu-dimages-annotees-pour-lanalyse-de-la-structure-de-mise-en-page-douvrages-anciens) (very diverse)
- [SynDoc](https://drive.google.com/file/d/1_goCKP5VeStjdDS0nGeZBPqPoLCMNyb6/view) (text lines, red)
- [IllusHisDoc](https://www.dropbox.com/s/bbpb9lzanjtj9f9/illuhisdoc.zip?dl%3D0) (illustrations, red)
- [DIVA-HisDB](https://diuf.unifr.ch/main/hisdoc/diva-hisdb.html) (various labels, red)

Some datasets that are accessible via IIIF (not supported yet):
- [HORAE](https://github.com/oriflamms/HORAE/)

Pixel-labeled datasets have not yet been used because they require manual intervention: the colours and the labels must be the same as those patched.


Tests
-----
For the moment, only integration tests of the classes of the _**`deep_learning_lab`**_ package have been implemented.
To execute them, simply do as follows:

    cd deep_learning/
    pytest



References
==========
[1] [Details about the specifications of the project.](https://github.com/theo-boi/deep-agora-doc)

[2] [S. Ares Oliveira, B.Seguin, and F. Kaplan, “dhSegment: A generic deep-learning approach for document segmentation”, in Frontiers in Handwriting Recognition (ICFHR), 2018 16th International Conference on, pp. 7-12, IEEE, 2018.](https://arxiv.org/abs/1804.10371)

[3] [K. Nikolaidou, M. Seuret, H. Mokayed and M. Liwicki, “A survey of historical document image datasets”, IJDAR 25, 305–338 (2022).](https://arxiv.org/abs/2203.08504)
