"""Orchestration module for data preparation and patching.

This module provides the Orchestrator class which is responsible for orchestrating the data preparation and patching
process for a given set of datasets. It contains a set of predefined datasets, and also allows for the creation of new
datasets through the specification of DataStructure objects. Additionally, the module provides the _analyze method
which is used for analyzing the statistics of the prepared datasets.

"""

from time import sleep
import os

from deep_learning_lab import logging
from deep_learning_lab.data_preparation.patch import DataStructure, AnnotationEncoder, DataPatcher


# Directory where the raw datasets are located
RAW_DATA_DIR = "raw_datasets" # "" if current directory
if RAW_DATA_DIR: os.makedirs(RAW_DATA_DIR, exist_ok= True)

_LOGGER = logging.getLogger(__name__)

# Constants used in many dataset paths
_BCS = "Baseline Competition - Simple Documents"
_BCC = "Baseline Competition - Complex Documents"


class Orchestrator:
    """A class to automate the preparation of training data."""

    def _pageDataStructure(name_dataset: str):
        """Create the parameters for a new DataStructure object that meets the PAGE specifications.

        Args:
            name_dataset (str): The name of the dataset.

        Returns:
            A DataStructure object representing the dataset structure that meets the PAGE specifications.

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
    
    
    def __init__(self, output_structure= None):
        """Constructor method for Orchestrator class.

        Args:
            output_structure (DataStructure, list, dict, str): A DataStructure object or the parameters to create one representing
                                                               the output directory structure of the patched datasets. If not
                                                               provided, a default DataStructure object will be created.

        """
        self.data_structures = list()   # Initialize an empty list to hold DataStructure objects
        self.sets_labels = list()   # Initialize an empty list to hold the labels of each dataset in data_structures
        if output_structure:
            self.output = self.__class__._implementDataStructure(output_structure)  # If an output_structure is provided, use it as the output attribute
        else:
            # Otherwise, create a default DataStructure object
            self.output = DataStructure(dir_data= "training_data", dir_images= "images", dir_labels= "labels")
        
        
    @property
    def data_specs(self) -> zip:
        """Returns a zipped list of data_structures and their labels.

        Returns:
            zip: A list of tuples containing a DataStructure object and its corresponding label.
        
        """
        return zip(self.data_structures, self.sets_labels)
    
    
    @classmethod
    def _implementDataStructure(cls, dataset):
        """Class method to create a DataStructure object corresponding to a given dataset.

        Args:
            dataset (DataStructure, list, dict, str): A DataStructure object or the parameters to create one.
                                                      It can also be a string representing a dataset name.

        Returns:
            DataStructure: The DataStructure object corresponding to the given dataset.

        Raises:
            NotImplementedError: If the provided dataset name is not recognized.
            ValueError: If the dataset is not a DataStructure, list, dict, or string.
        
        """
        if isinstance(dataset, DataStructure):
            return dataset
        
        elif isinstance(dataset, list):
            assert len(dataset) == 3   # Ensure the list has 3 elements
            assert sum(map(lambda path: not isinstance(path, str), dataset)) == 0
            return DataStructure(*dataset)
        
        elif isinstance(dataset, dict):
            return DataStructure(**dataset)

        elif isinstance(dataset, str):
            dataset = dataset.lower()
            if dataset in cls.DATASETS:
                return cls.DATASETS[dataset]
            else:
                raise NotImplementedError(f"'{dataset}' dataset have not been implemented.")

        else:
            value_error = ValueError(f"Dataset must be either a dictionary of DataStructure parameters or a dataset name")
            _LOGGER.error(f"{repr(value_error)}: {str(dataset)}")
            raise value_error
         
        
    @staticmethod
    def _analyze(stats_files: dict, verbose: int = 0) -> dict:
        """Print dataset statistics.

        Args:
            stats_files (dict): Dictionary containing the statistics to print.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            dict: Statistics files.
        
        """
        n_unique_labels = stats_files.pop("N_UNIQUE_LABELS")
        n_labels = stats_files.pop("N_LABELS")
        number_labels = stats_files.pop("LABELS")
        
        if verbose:
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
    
    
    def ingestDatasets(self, datasets= [], add_defaults: bool = True) -> None:
        """Load datasets and add them to the list of data structures. If `add_defaults` is True or no datasets are specified,
         add all default datasets to the list of data structures.

        Args:
            datasets (iterable, optional): An iterable of datasets to load. Defaults to [].
            add_defaults (bool, optional): If True, add all default datasets to the list of data structures. Defaults to True.
        
        """

        # Check if output is an instance of DataStructure, if not, create a DataStructure object
        self.output = self.__class__._implementDataStructure(self.output)
        
        if not datasets or add_defaults:
            # Add default datasets if none specified or `add_defaults` is True
            datasets.extend(
                [
                    default_structure
                    for default_name, default_structure in self.DATASETS.items()
                    if default_name not in datasets
                ]
            )

        # Load and add all datasets to the list of data structures
        for data in datasets:
            self.data_structures.append(self.__class__._implementDataStructure(data))
    
    
    def ingestLabels(self, uniform_set_labels= [], prompt: bool = True) -> None:
        """Allow the users to extract and select the annotations they want to convert into masks from the dataset.
        The selected annotations can be either a single one or an arrangement of several.
        The extraction and selection is done within the console.

        Args:
            uniform_set_labels (iterable, optional): An iterable set of labels to select. Defaults to [].
            prompt (bool, optional): If True, prompt the user to select annotations. Defaults to True.
        
        """
        
        # Create list of labels for each data structure
        self.sets_labels = [uniform_set_labels.copy() for _ in range(len(self.data_structures))]
        
        for data_structure, set_labels in self.data_specs:
            # Clean absent labels from the data structure's annotations
            ae = AnnotationEncoder(data_structure.dir_annotations)
            ae.cleanAbsentLabels(set_labels)

            if prompt:
                # Prompt user to select annotations
                print(f"ANNOTATION SELECTION FOR `{data_structure}`")
                if set_labels:
                    print(f"{', '.join(set_labels)} have been pre-selected")
                print()

                valid = False
                while not valid:
                    # Choose labels and analyze statistics
                    chosen_labels = ae.chooseLabels(preselection= set_labels)
                    print()
                    self._analyze(ae.calculateStatistics(chosen_labels), verbose= 1)

                    # Allow user to modify selection
                    decision = input(f"Modify selection for {str(data_structure)} ([y]/n)? ").lower()
                    print()
                    valid = (decision == 'n' or decision == 'no')
                set_labels[:] = chosen_labels
    
    
    def validate(self, auto_yes: bool = False, verbose: bool = True):
        """Validate the labels and datasets by calculating statistics for the selected annotations.

        Args:
            auto_yes (bool, optional): automatically validate the labels and datasets without user confirmation.
            verbose (bool, optional): the level of verbosity. If True, will print the statistics of each dataset.
        
        """
        if not auto_yes:
            _LOGGER.warning("User is not asked to validate by himself")

        if verbose:
            # Calculate statistics for the selected annotations and print them to the console
            for data_structure, set_labels in self.data_specs:
                ae = AnnotationEncoder(data_structure.dir_annotations)
                print(f"VALIDATION OF `{data_structure}`")
                self.__class__._analyze(ae.calculateStatistics(set_labels), verbose)
                print()
            
        if not auto_yes:
            decision = input("Validate ([y]/n)? ").lower()
            print()
            if decision == 'n' or decision == 'no':
                _LOGGER.info("User decided not to validate their labels or datasets")
                raise Exception("Try to fine-tuning the labels in ingestLabels or rethinking your datasets.")
        
        
    def preprocess(self, resize: tuple= (1188, 841), overwrite: bool = True, verbose: int = 0) -> None:
        """Convert the dataset located at src to a compatible one located at dest.

        Args:
            resize (tuple): the new dimensions of the images after resizing. Default is (1188, 841).
            overwrite (bool, optional): if True, the preprocessed dataset will overwrite the existing patched dataset. Default is True.
            verbose (int, optional): the level of verbosity. 0 is none, 1 is message only, 2 is message and progress bar,
                                     3 is message and debug, and 4+ is all.

        Todo:
            Keep AnnotationReader/Writer of dhSegment.data in mind for specific types of datasets

        """

        # Determine which messages to print based on verbose level
        print_msg = (verbose >= 1)
        print_progress = (verbose == 2 or verbose >= 4)
        print_debug = (verbose >= 3)

        # Loop through data structures and their labels
        for data_structure, set_labels in self.data_specs:
            if set_labels:
                # Prepare message for patching
                msg = f"PATCHING `{data_structure}` WITH {'`'+', '.join(set_labels)+'`' if set_labels else 'NO'} SPECIFIED LABEL{'S' if len(set_labels)>1 else ''}"
                _LOGGER.info(msg)
                if print_msg:
                    print(msg)
                    if print_progress: sleep(0.25)

                # Create DataPatcher object and patch the data
                patcher = DataPatcher(data_structure, self.output)
                patcher.patch(
                    set_labels,
                    resize,
                    overwrite,
                    print_progress,
                    print_debug
                )

                if print_progress:
                    sleep(0.25)
                    print()
            else:
                # Prepare message for skipping patching
                msg = f"NOT PATCHING `{data_structure}` AS THERE ARE NO ANNOTATIONS FOR THESE LABELS"
                _LOGGER.info(msg)
                if print_msg:
                    print(msg)
                    print()
