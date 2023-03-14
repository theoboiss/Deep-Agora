"""
=======================================
Automation of training data preparation
=======================================

This module offers a simple pipeline designed to build training data under the right
format for dhSegment models.

For the moment, it can only patch datasets under the PAGE format.

"""

import os
from collections import defaultdict
from time import sleep

from data_preparation.patch import DataStructure, AnnotationEncoder, DataPatcher


RAW_DATA_DIR = "raw_datasets" # "" if current directory


_BCS = "Baseline Competition - Simple Documents"
_BCC = "Baseline Competition - Complex Documents"


class Orchestrator:
    
    def _pageDataStructure(name_dataset):
        """
        Create the parameters for a new DataStructure object that meets the PAGE specifications.
        """
        return DataStructure(dir_data= name_dataset, dir_images= "", dir_annotations= "page")

    DATASETS = {
        'reid': DataStructure(
            dir_data= os.path.join(RAW_DATA_DIR, "REID2019"),
            dir_images= "REID2019_ExampleSet",
            dir_annotations= "REID2019_ExampleSet"
        ),
        'fcr': _pageDataStructure(os.path.join(RAW_DATA_DIR,"FCR_500","data")),
        'bcs_a': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"ABP_FirstTestCollection")),
        'bcs_b': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"Bohisto_Bozen_SetP")),
        'bcs_e': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"EPFL_VTM_FirstTestCollection")),
        'bcs_h': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"HUB_Berlin_Humboldt")),
        'bcs_n': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"NAF_FirstTestCollection")),
        'bcs_s': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"StAM_Marburg_Grimm_SetP")),
        'bcs_u': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"UCL_Bentham_SetP")),
        'bcs_un': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCS,"unibas_e-Manuscripta")),
        'bcc_a': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"ABP_FirstTestCollection")),
        'bcc_b': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"Bohisto_Bozen_SetP")),
        'bcc_bh': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"BHIC_Akten")),
        'bcc_e': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"EPFL_VTM_FirstTestCollection")),
        'bcc_h': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"HUB_Berlin_Humboldt")),
        'bcc_n': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"NAF_FirstTestCollection")),
        'bcc_s': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"StAM_Marburg_Grimm_SetP")),
        'bcc_u': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"UCL_Bentham_SetP")),
        'bcc_un': _pageDataStructure(os.path.join(RAW_DATA_DIR,_BCC,"unibas_e-Manuscripta")),
    }
    
    
    def __init__(self, output= None):
        self.data_structures = list()
        self.sets_labels = list()
        if output:
            self.output = output
        else:
            self.output = DataStructure(dir_data= "data", dir_images= "images", dir_labels= "labels")
        
        
    @property
    def data_specs(self):
        return zip(self.data_structures, self.sets_labels)
    
    
    @classmethod
    def _implementDataStructure(cls, dataset):
        """
        Create the DataStructure object corresponding to dataset.
        """
        if isinstance(dataset, DataStructure):
            return dataset
        
        elif isinstance(dataset, dict):
            return DataStructure(**dataset)

        elif isinstance(dataset, str):
            dataset = dataset.lower()
            if dataset in cls.DATASETS:
                return cls.DATASETS[dataset]
            else:
                raise NotImplementedError(f"'{dataset}' dataset have not been implemented.")

        else:
            raise ValueError(f"dataset must be either a dictionary of DataStructure parameters or a dataset name")
         
        
    @staticmethod
    def _analyze(stats_files, verbose= 0):
        n_unique_labels = stats_files.pop("N_UNIQUE_LABELS")
        n_labels = stats_files.pop("N_LABELS")
        number_labels = stats_files.pop("LABELS")
        
        if number_labels:
            if verbose >= 1:
                print(', '.join((str(number)+' '+label for label, number in number_labels.items())))

                if verbose >= 2:
                    for file, stats in stats_files.items():
                        filename = os.path.basename(file)

                        if stats['N_LABELS']:
                            print(f"\t{filename} ({stats['N_LABELS']})")

                            if verbose >= 3:
                                for label, number in stats['LABELS'].items():
                                    if number:
                                        print(f"\t\t{number} {label}")

                        elif verbose >=4:
                            print(f"\t{filename} ({stats['N_LABELS']})")
        
        else:
            print("No label")
        return dict(stats_files, N_UNIQUE_LABELS= n_unique_labels, N_LABELS= n_labels, LABELS= number_labels)
    
    
    def ingestDatasets(self, datasets= [], add_default= True):
        self.output = self.__class__._implementDataStructure(self.output)
        
        if not datasets or add_default:
            datasets.extend(self.DATASETS.values())
        for data in datasets:
            try:
                self.data_structures.append(self.__class__._implementDataStructure(data))
            except Exception as e:
                print(e)
    
    
    def ingestLabels(self, uniform_set_labels= [], prompt= True):
        """
        Allow the users to extract and select the annotations they want to convert into masks from the dataset.
        The selected annotations can be either a single one or an arrangement of several.
        The extraction and selection is done within the console.
        """
        self.sets_labels = [uniform_set_labels.copy() for _ in range(len(self.data_structures))]
        
        for data_structure, set_labels in self.data_specs:
            ae = AnnotationEncoder(data_structure.dir_annotations)
            ae.cleanAbsentLabels(set_labels)

            if prompt:
                print(f"ANNOTATION SELECTION FOR {str(data_structure).upper()}")
                if set_labels:
                    print(f"{', '.join(set_labels)} have been pre-selected")
                print()

                valid = False
                while not valid:
                    chosen_labels = ae.chooseLabels(preselection= set_labels)
                    print()

                    self._analyze(ae.calculateStatistics(chosen_labels), verbose= 1)

                    decision = input(f"Modify selection for {str(data_structure)} ([y]/n)? ").lower()
                    print()
                    valid = (decision == 'n' or decision == 'no')
                set_labels[:] = chosen_labels
    
    
    def validate(self, auto_yes= False, verbose= 0):
        """
        
        
        verbose: the level of verbose. 0 is none, 1 is message only, 2 is message and progress bar, 3 is message and debug, and 4+ is all.
        """
        if not verbose:
            print("Warning: User will not be asked to validate by himself.")
        
        for data_structure, set_labels in self.data_specs:
            ae = AnnotationEncoder(data_structure.dir_annotations)
            if verbose:
                print(f"VALIDATION OF {str(data_structure).upper()}")
                self.__class__._analyze(ae.calculateStatistics(set_labels), verbose)
                print()
            
        if not auto_yes and verbose:
            decision = input("Validate ([y]/n)? ").lower()
            print()
            if decision == 'n' or decision == 'no':
                raise Exception("Try to fine-tuning the labels in ingestLabels or rethinking your datasets.")
        
        
    def preprocess(self, verbose= 0):
        """
        Convert the dataset located at src to a compatible one located at dest.

        names_labels: the name of the labels to convert
        verbose: the level of verbose. 0 is none, 1 is message only, 2 is message and progress bar, 3 is message and debug, and 4+ is all.
        """
        print_msg = (verbose >= 1)
        print_progress = (verbose == 2 or verbose >= 4)
        print_debug = (verbose >= 3)

        for data_structure, set_labels in self.data_specs:
            if set_labels:
                if print_msg:
                    print(f"PATCHING {str(data_structure).upper()} WITH {', '.join(set_labels).upper() if set_labels else 'NO'} SPECIFIED LABEL{'S' if len(set_labels)>1 else ''}")
                    if print_progress: sleep(0.25)
                patcher = DataPatcher(data_structure, self.output)
                patcher.patch(
                    names_labels= set_labels,
                    size_img= (637, 900),
                    verbose= print_progress,
                    debug_annotations= print_debug
                )
                if print_progress:
                    sleep(0.25)
                    print()
                # Keep AnnotationReader/Writer in mind for specific types of datasets
