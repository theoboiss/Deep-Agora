"""
=======================================
Automation of training data preparation
=======================================

This module offers a simple pipeline designed to build training data under the right
format for dhSegment models.

For the moment, it can only patch datasets under the PAGE format.

"""

import os
from time import sleep
import csv

from training_data.patch import DataStructure, AnnotationEncoder, DataPatcher


RAW_DATA_DIR = "raw_datasets" # "" if current directory


_BCS = "Baseline Competition - Simple Documents"
_BCC = "Baseline Competition - Complex Documents"

def _page_data_dict(name_dataset):
    """
    Create the parameters for a new DataStructure object that meets the PAGE specifications.
    """
    return {'dir_data' : name_dataset, 'dir_images' : "", 'dir_annotations' : "page"}

DATASETS = {
    'reid': {
        'dir_data' : os.path.join(RAW_DATA_DIR, "REID2019"),
        'dir_images' : "REID2019_ExampleSet",
        'dir_annotations' : "REID2019_ExampleSet"
    },
    'fcr': _page_data_dict(os.path.join(RAW_DATA_DIR,"FCR_500","data")),
    'bcs_a': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"ABP_FirstTestCollection")),
    'bcs_b': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"Bohisto_Bozen_SetP")),
    'bcs_e': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"EPFL_VTM_FirstTestCollection")),
    'bcs_h': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"HUB_Berlin_Humboldt")),
    'bcs_n': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"NAF_FirstTestCollection")),
    'bcs_s': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"StAM_Marburg_Grimm_SetP")),
    'bcs_u': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"UCL_Bentham_SetP")),
    'bcs_un': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCS,"unibas_e-Manuscripta")),
    'bcc_a': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"ABP_FirstTestCollection")),
    'bcc_b': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"Bohisto_Bozen_SetP")),
    'bcc_bh': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"BHIC_Akten")),
    'bcc_e': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"EPFL_VTM_FirstTestCollection")),
    'bcc_h': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"HUB_Berlin_Humboldt")),
    'bcc_n': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"NAF_FirstTestCollection")),
    'bcc_s': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"StAM_Marburg_Grimm_SetP")),
    'bcc_u': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"UCL_Bentham_SetP")),
    'bcc_un': _page_data_dict(os.path.join(RAW_DATA_DIR,_BCC,"unibas_e-Manuscripta")),
    '_data_': {'dir_data': "data", 'dir_images': "images",'dir_labels': "labels"},
}



def implementDataStructure(name_dataset):
    """
    Create the DataStructure object corresponding to name_dataset.
    """
    name_dataset = name_dataset.lower()
    
    if name_dataset in DATASETS:
        data = DataStructure(**DATASETS[name_dataset])
    else:
        raise NotImplementedError(f"'{name_dataset}' dataset have not been implemented.")
    return data


def statisticsOf(original_dataset, names_labels, details= 0):
    data = implementDataStructure(original_dataset)
    
    ae = AnnotationEncoder(data.dir_annotations)
    stats_files = ae.calculateStatistics(names_labels)
    
    n_unique_labels = stats_files.pop("N_UNIQUE_LABELS")
    n_labels = stats_files.pop("N_LABELS")
    number_labels = stats_files.pop("LABELS")
    if details >= 0:
        if n_labels:
            print(f"{original_dataset.upper()} dataset contains", end= '')
            for label, number in number_labels.items():
                print(f" {number} {label},", end= '')
            if details >= 1:
                print()
                for file, stats in stats_files.items():
                    filename = os.path.basename(file)
                    if stats['N_LABELS']:
                        print(f"\t{filename} ({stats['N_LABELS']}){':' if stats['LABELS'] and details >= 3 else ''}")
                        if details >= 2:
                            for label, number in stats['LABELS'].items():
                                if number:
                                    print(f"\t\t{number} {label}")
                    elif details >=3:
                        print(f"\t0 label in {filename}")
        else:
            print(f"{original_dataset.upper()} dataset does not contain" +
                  f" any {', '.join(names_labels)} label{'s' if len(names_labels)>1 else ''}")


def selectAnnotationsFrom(original_dataset):
    """
    Allow the users to extract and select the annotations they want to convert into masks from the dataset.
    The selected annotations can be either a single one or an arrangement of several.
    The extraction and selection is done within the console.
    """
    data = implementDataStructure(original_dataset)
    chosen_annotations = []
    
    print(f"ANNOTATION SELECTION FOR THE {original_dataset.upper()} DATASET")
    print()
    patcher = DataPatcher(data)

    valid = False
    while not valid:
        chosen_annotations.append(patcher.annotationsAnalysis())
        decision = input("Continue ([y]/n)? ").lower()
        print()
        if decision == 'n' or decision == 'no':
            valid = True
    return chosen_annotations


def annotationsToMasks(src, dest= "_data_", names_labels= None, verbose= 0):
    """
    Convert the dataset located at src to a compatible one located at dest.
    
    names_labels: the name of the labels to convert
    verbose: the level of verbose. 0 is none, 1 is message only, 2 is message and progress bar, 3 is message and debug, and 4+ is all.
    """
    print_msg = (verbose >= 1)
    print_progress = (verbose == 2 or verbose >= 4)
    print_debug = (verbose >= 3)
    
    src_data = implementDataStructure(src)
    dest_data = implementDataStructure(dest)

    if print_msg:
        print(f"PATCHING {src.upper()} WITH {', '.join(names_labels) if names_labels else 'NO'} SPECIFIED LABEL{'S' if len(names_labels)>1 else ''}")
        if print_progress: sleep(0.25)
    patcher = DataPatcher(src_data, dest_data)
    patcher.patch(
        size_img= (637, 900),
        names_labels= names_labels,
        verbose= print_progress,
        debug_annotations= print_debug
    )
    if print_progress:
        sleep(0.25)
        print()
    # Keep AnnotationReader/Writer in mind for specific types of datasets
