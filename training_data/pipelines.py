import os

from training_data.patch import DataStructure, DataPatcher


def data_structure_page(name_dataset):
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
        data = data_structure_page(os.path.join("FCR_500", "data"))
        
    elif name_dataset == '_test_fcr_':
        data = data_structure_page(os.path.join("FCR", "data"))


    elif name_dataset == 'bcs_a':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "ABP_FirstTestCollection"))

    elif name_dataset == 'bcs_b':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "Bohisto_Bozen_SetP"))

    elif name_dataset == 'bcs_e':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "EPFL_VTM_FirstTestCollection"))

    elif name_dataset == 'bcs_h':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "HUB_Berlin_Humboldt"))

    elif name_dataset == 'bcs_n':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "NAF_FirstTestCollection"))

    elif name_dataset == 'bcs_s':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "StAM_Marburg_Grimm_SetP"))

    elif name_dataset == 'bcs_u':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "UCL_Bentham_SetP"))

    elif name_dataset == 'bcs_un':
        data = data_structure_page(os.path.join("Baseline Competition - Simple Documents", "unibas_e-Manuscripta"))


    elif name_dataset == '_data_':
        dir_original_data = 'data'

        data = DataStructure(
            dir_data= dir_original_data,
            dir_images= "images",
            dir_labels= "labels"
        )
    
    else:
        raise NotImplementedError
    return data

def annotationsToMasks(src, dest= "_data_", names_labels= None, verbose= False):
    src_data = implementDataStructure(src)
    dest_data = implementDataStructure(dest)

    dp = DataPatcher(src_data, dest_data)
    dp.patch(names_labels= names_labels, debug= verbose)
    # Keep AnnotationReader/Writer in mind for specific types of datasets
