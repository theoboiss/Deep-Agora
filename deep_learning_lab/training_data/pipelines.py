"""
=======================================
Automation of training data preparation
=======================================

This module offers a pipeline designed to build training data under the right
format for dhSegment models.

Its interactive functionnality allows the user to design diverse training
dataset according to the elements of content they want to extract.
For the moment, it can only patch datasets under the PAGE format.

"""

import os
from time import sleep

from training_data.patch import DataStructure, DataPatcher


RAW_DATA_DIR = "raw_datasets" # "" if deactivated


def page_data_structure(name_dataset):
    dir_original_data = name_dataset
    data = DataStructure(
        dir_data= dir_original_data,
        dir_images= "",
        dir_annotations= "page"
    )
    return data

def implementDataStructure(name_dataset):
    name_dataset = name_dataset.lower()
    
    if name_dataset == 'fcr':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "FCR_500",
            "data"
        ))

    elif name_dataset == 'bcs_a':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "ABP_FirstTestCollection"
        ))
    elif name_dataset == 'bcs_b':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "Bohisto_Bozen_SetP"
        ))
    elif name_dataset == 'bcs_e':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "EPFL_VTM_FirstTestCollection"
        ))
    elif name_dataset == 'bcs_h':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "HUB_Berlin_Humboldt"
        ))
    elif name_dataset == 'bcs_n':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "NAF_FirstTestCollection"
        ))
    elif name_dataset == 'bcs_s':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "StAM_Marburg_Grimm_SetP"
        ))
    elif name_dataset == 'bcs_u':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "UCL_Bentham_SetP"
        ))
    elif name_dataset == 'bcs_un':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Simple Documents",
            "unibas_e-Manuscripta"
        ))

    elif name_dataset == 'bcc_a':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "ABP_FirstTestCollection"
        ))
    elif name_dataset == 'bcc_b':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "Bohisto_Bozen_SetP"
        ))
    elif name_dataset == 'bcc_bh':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "BHIC_Akten"
        ))
    elif name_dataset == 'bcc_e':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "EPFL_VTM_FirstTestCollection"
        ))
    elif name_dataset == 'bcc_h':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "HUB_Berlin_Humboldt"
        ))
    elif name_dataset == 'bcc_n':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "NAF_FirstTestCollection"
        ))
    elif name_dataset == 'bcc_s':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "StAM_Marburg_Grimm_SetP"
        ))
    elif name_dataset == 'bcc_u':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "UCL_Bentham_SetP"
        ))
    elif name_dataset == 'bcc_un':
        data = page_data_structure(os.path.join(
            RAW_DATA_DIR,
            "Baseline Competition - Complex Documents",
            "unibas_e-Manuscripta"
        ))

    elif name_dataset == 'reid':
        dir_original_data = os.path.join(
            RAW_DATA_DIR,
            "REID2019"
        )
        data = DataStructure(
            dir_data= dir_original_data,
            dir_images= "REID2019_ExampleSet",
            dir_annotations= "REID2019_ExampleSet"
        )

    elif name_dataset == '_data_':
        dir_original_data = "data"
        data = DataStructure(
            dir_data= dir_original_data,
            dir_images= "images",
            dir_labels= "labels"
        )
    
    else:
        raise NotImplementedError(f"{name_dataset} have not been implemented.")
    return data

def selectAnnotationsFrom(original_dataset):
    data = implementDataStructure(original_dataset)
    chosen_annotations = []
    
    print(f"ANNOTATION SELECTION FOR THE {original_dataset.upper()} DATASET")
    print()
    dp = DataPatcher(data)

    valid = False
    while not valid:
        chosen_annotations.append(dp.annotationsAnalysis())
        decision = input("Continue ([y]/n)? ").lower()
        print()
        if decision == 'n' or decision == 'no':
            valid = True
    return chosen_annotations

def annotationsToMasks(src, dest= "_data_", names_labels= None, verbose= True, debug= False):
    src_data = implementDataStructure(src)
    dest_data = implementDataStructure(dest)

    if verbose:
        print(f"PATCHING {src.upper()} WITH '{', '.join(names_labels) if names_labels else 'NO'}' SPECIFIED LABEL{'S' if len(names_labels)>1 else ''}")
        sleep(0.25)
    dp = DataPatcher(src_data, dest_data)
    dp.patch(names_labels= names_labels, verbose= debug, debug= debug)
    if debug:
        sleep(0.25)
        print()
    # Keep AnnotationReader/Writer in mind for specific types of datasets
