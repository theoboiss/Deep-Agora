"""Module for semantic segmentation model training and inference.

This module contains classes for training and predicting with semantic segmentation models.
The `Trainer` class allows users to setup, fine-tune, and train a semantic segmentation model using the `dhSegment` library.
The `Predictor` class provides functionality for performing inference on a trained model.

Classes:
    ModelUser: Abstract class designed to share model and data architectures between Trainer and Predictor.
    Trainer: A class implementing the dhSegment Trainer interface designed to setup, fine-tune, and train a semantic
             segmentation model.
    Predictor: A class to perform inference on a semantic segmentation model using PredictProcess of dhSegment.

Usage:
    Import the required class from the module and create an object with the necessary parameters to use the functions provided
    by the class.

For more information about each class and its parameters, please refer to the class docStrings.

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC
import os, glob, cv2, shutil

from dh_segment_torch.config import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.training import Trainer as dhTrainer
from dh_segment_torch.inference import PredictProcess

from deep_learning_lab import logging


_LOGGER = logging.getLogger(__name__)

_DESC_PROGRESSBAR_POSTPROCESS = "Post-processing predictions "



class ModelUser(ABC):
    """Abstract class designed to share model and data locations."""

    def __init__(self, labels, input_dir: str, workdir: str = "results"):
        """Initializes the ModelUser object.

        Args:
            labels (iterable): Iterable of labels used in the model.
            input_dir (str): Name of the input directory.
            workdir (str, optional): Name of the working directory. Defaults to "results".

        """
        self.workdir = os.path.join(workdir, '_'.join(labels))
        os.makedirs(self.workdir, exist_ok= True)
        self.data_dir = os.path.join(self.workdir, input_dir)
        os.makedirs(self.data_dir, exist_ok= True)
        self.model_dir = os.path.join(self.workdir, "model")
        
        
    def __str__(self) -> str:
        """Returns the string representation of the ModelUser object."""
        return self.__class__.__name__ + ": " + self.__dict__

        
        
class Trainer(ModelUser):
    """A class implementing the dhSegment Trainer interface designed to setup, fine-tune, and train a semantic segmentation model."""
    
    def __init__(self, labels, workdir: str = "results", input_dir: str = "training_data", train_ratio: float = 0.80, val_ratio: float = 0.10):
        """Initialize the Trainer class instance.

        Args:
            labels (iterable): A set of label names used to choose the appropriate resources in the workdir.
            workdir (str): The directory name of the resources and outputs. Default is "results".
            input_dir (str): The directory name of the patched dataset in the workdir. Default is "training_data".
            train_ratio (float): The ratio of data to use for training (between 0.0 and 1.0). Default is 0.80.
            val_ratio (float): The ratio of data to use for validation (between 0.0 and 1.0-train_ratio). Default is 0.10.

        """
        super().__init__(labels, input_dir, workdir)
        
        self.tensorboard_dir = os.path.join(self.workdir, 'tensorboard', 'log')
        self._setupEnvironment(train_ratio, val_ratio)

        
    def _setupEnvironment(self, train_ratio: float, val_ratio: float) -> dict:
        """Sets up the environment for training a machine learning model.

        Args:
            train_ratio (float): Ratio of data to be used for training.
            val_ratio (float): Ratio of data to be used for validation.

        Returns:
            params (dict): A dictionary containing the parameters used to set up the environment.

        """

        # Set up the data path and splitting ratio
        params = {
            'data_path' : self.data_dir, # Path to write the data
            'data_splitter': {'train_ratio': train_ratio, 'val_ratio': val_ratio, 'test_ratio': 1.00-train_ratio-val_ratio}, # splitting ratio of the data
        }
        
        
        ## Process parameters

        # Set default values for parameters
        relative_path = params.pop("relative_path", True)

        params.setdefault("labels_dir", os.path.join(self.data_dir, "labels"))
        labels_dir = params.get("labels_dir")

        params.setdefault("images_dir", os.path.join(self.data_dir, "images"))
        images_dir = params.get("images_dir")

        params.setdefault("csv_path", os.path.join(self.data_dir, "data.csv"))

        # Get parameters for data splitter
        data_splitter_params = params.pop("data_splitter", None)
        train_csv_path = params.pop("train_csv", os.path.join(self.data_dir, "train.csv"))
        val_csv_path = params.pop("val_csv", os.path.join(self.data_dir, "val.csv"))
        test_csv_path = params.pop("test_csv", os.path.join(self.data_dir, "test.csv"))

        
        ## List labels and images in CSV file

        # Get a list of label and image files
        labels_list = sorted(glob.glob(os.path.join(labels_dir, '*.*')))
        images_list = sorted(glob.glob(os.path.join(images_dir, '*.*')))

        # Create a dataframe containing the file paths for labels and images
        data = pd.DataFrame({'image': images_list, 'label': labels_list})
        data.to_csv(params['csv_path'], header=False, index=False)

        # If relative path is True, update the paths in the dataframe to be relative to the data directory
        if relative_path:
            data['image'] = data['image'].apply(lambda path: os.path.join("images", os.path.basename(path)))
            data['label'] = data['label'].apply(lambda path: os.path.join("labels", os.path.basename(path)))

        ## Divide data set in train, validation and test sets

        # If data splitter parameters are provided, split the data into train, validation, and test sets
        if data_splitter_params:
            data_splitter = DataSplitter.from_params(data_splitter_params)
            data_splitter.split_data(data, train_csv_path, val_csv_path, test_csv_path)
            
        return params
       
    
    def _setupTraining(self, batch_size: int, epochs: int, learning_rate: float, gamma_exp_lr: float,
                       evaluate_every_epoch: int, val_patience: int, output_size: int, repeat_dataset: int) -> dict:
        """Configure training parameters and return as a dictionary.

        Args:
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train for.
            learning_rate (float): The initial learning rate for the optimizer.
            gamma_exp_lr (float): The gamma value to use for the exponential learning rate scheduler.
            evaluate_every_epoch (int): Number of epochs to wait before evaluating the model on the validation dataset.
            val_patience (int): Number of epochs to wait before stopping training if the validation score does not improve.
            output_size (int): The output size for the fixed size resize transform.
            repeat_dataset (int): The number of times to repeat the training dataset.

        Returns:
            dict: A dictionary of training parameters.

        """
        params = {
            "color_labels": {"label_json_file": os.path.join(self.data_dir, "classfile.json")}, # Color labels produced before
            "train_dataset": {
                "type": "image_csv", # Image csv dataset
                "csv_filename": os.path.join(self.data_dir, "train.csv"),
                "base_dir": self.data_dir,
                "repeat_dataset": repeat_dataset, # Repeat 4 times the data since we have little
                "compose": {"transforms": [{"type": "fixed_size_resize", "output_size": output_size}]} # Resize to a fixed size, could add other transformations.
            },
            "val_dataset": {
                "type": "image_csv", # Validation dataset
                "csv_filename": os.path.join(self.data_dir, "val.csv"),
                "base_dir": self.data_dir,
                "compose": {"transforms": [{"type": "fixed_size_resize", "output_size": output_size}]}
            },
            "model": { # Model definition, original dhSegment
                "encoder": "resnet50", 
                "decoder": {
                    "decoder_channels": [512, 256, 128, 64, 32],
                    "max_channels": 512
                }
            },
            "metrics": [['miou', 'iou'], ['iou', {"type": 'iou', "average": None}], 'precision'], # Metrics to compute
            "optimizer": {"lr": learning_rate}, # Learning rate
            "lr_scheduler": {"type": "exponential", "gamma": gamma_exp_lr}, # Exponential decreasing learning rate
            "val_metric": "+miou", # Metric to observe to consider a model better than another, the + indicates that we want to maximize
            "early_stopping": {"patience": val_patience}, # Number of validation steps without increase to tolerate, stops if reached
            "model_out_dir": self.model_dir, # Path to model output
            "num_epochs": epochs, # Number of epochs for training
            "evaluate_every_epoch": evaluate_every_epoch, # Number of epochs between each validation of the model
            "batch_size": batch_size, # Batch size (to be changed if the allocated GPU has little memory)
            "num_data_workers": 0,
            "track_train_metrics": False,
            "loggers": [
                {"type": 'tensorboard', "log_dir": self.tensorboard_dir, "log_every": 5, "log_images_every": 10}, # Tensorboard logging
            ]
        }
        return params
        
        
    def train(self, batch_size: int = 1, epochs: int = 1, learning_rate: float = 1e-1, gamma_exp_lr: float = 1.0,
              evaluate_every_epoch: int = 1, val_patience: int = 1, repeat_dataset: int = 1, output_size: int = 1e6) -> None:
        """Train the model using the specified hyperparameters.

        Args:
            batch_size (int): The number of samples to use in each training batch.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the optimizer.
            gamma_exp_lr (float): The exponential decay rate of the learning rate.
            evaluate_every_epoch (int): The frequency of validation evaluation in epochs.
            val_patience (int): The number of epochs to wait before early stopping if validation loss does not improve.
            repeat_dataset (int): The number of times to repeat the dataset per epoch.
            output_size (int): The maximum size of the output prediction.

        Returns:
            dict: A dictionary of the training parameters.

        """

        # Setup training parameters
        params = self._setupTraining(batch_size, epochs, learning_rate, gamma_exp_lr, evaluate_every_epoch,
                                     val_patience, output_size, repeat_dataset)

        # Create a trainer instance
        trainer = dhTrainer.from_params(params)

        # Train the model
        _LOGGER.info(f"Starting training of model in {self.model_dir}")
        trainer.train()
        _LOGGER.info(f"Model trained and serialized")
        
        
        
def _n_colors(n: int) -> list:
    """Returns a list of n random RGB colors generated with a constant seed.

    Args:
        n (int): The number of colors to generate.

    Returns:
        list: A list of n RGB colors represented as tuples of three integers in the range [0, 255].

    Raises:
        AssertionError: If n is falsy (e.g., zero).
    
    """
    assert n, "n must be a positive integer"
    
    # Initialize the list of colors with black
    colors = [(0, 0, 0)]
    
    # Initialize a random number generator with a constant seed
    rng = np.random.default_rng(0);
    r = int(rng.random() * 256)
    g = int(rng.random() * 256)
    b = int(rng.random() * 256)
    
    # Compute the step size for the subsequent colors
    step = 256 / n
    
    # Generate n-1 additional colors
    for _ in range(n-1):
        # Increment the color values by the step size
        r += step
        g += step
        b += step
        
        # Wrap the color values around if they exceed 255
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        
        # Append the new color to the list
        colors.append((r, g, b))
    return colors



class Predictor(ModelUser):
    """A class to perform inference on a semantic segmentation model using PredictProcess of dhSegment."""
    
    def __init__(self, labels, input_dir: str = 'inference_data', output_dir: str = None, output_size: tuple = None,
                 from_csv: str = None, reset_from_csv: bool = True):
        """Initialize the Predictor class instance.

        Args:
            labels (iterable): An iterable of label names.
            input_dir (str, optional): The path to the input data directory. Defaults to 'inference_data'.
            output_dir (str, optional): The path to the output directory. If not provided, the directory 'predictions' will be created within the workdir. Defaults to None.
            output_size (tuple, optional): The output size of the segmentation model. Defaults to None.
            from_csv (str, optional): The name of the CSV file containing the input data. Defaults to None.
            reset_input (bool, optional): Whether or not to delete the contents of the input directory before copying new input data. Defaults to True.
        
        """
        super().__init__(labels, input_dir)
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(self.workdir, 'predictions')
            
        # Set number of classes, colors, and output size
        self.num_classes = len(labels)+1
        self.colors = _n_colors(self.num_classes)
        self.output_size = output_size
        self.results = None
        
        # Convert input data from CSV to folder
        if from_csv:
            self._dataCSVToFolder(os.path.join(self.workdir, from_csv), reset_from_csv)
            
        # Create output directory if it does not exist
        if not os.path.exists(self.output_dir):
            _LOGGER.info(f"{self.output_dir} created")
        os.makedirs(self.output_dir, exist_ok= True)
        
        
    @staticmethod
    def probasToMaps(probas: np.ndarray, id_class: int) -> np.ndarray:
        """Converts the predicted probability maps to a map of classes.

        Args:
            probas (ndarray): Array of predicted probability maps.
            id_class (int): Index of the class to convert to map.

        Returns:
            ndarray: Map of the specified class.

        """
        # Get the probability map for the specified class and round the values to integers
        maps_probas_class = probas[id_class]
        map_probas = np.around(maps_probas_class*255)

        # Cast to uint8 and return the map
        return map_probas.astype('uint8')
    
    
    @staticmethod
    def _findContours(labels: np.ndarray) -> list:
        """Finds contours in an image.

        Args:
            labels (ndarray): Image to find contours in.

        Returns:
            list: List of contours.

        """
        # Convert the image to grayscale and threshold to create a binary image
        grey_labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
        _, thresh_labels = cv2.threshold(grey_labels, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    
    @classmethod
    def __drawRegions(cls, labels: np.ndarray, image: np.ndarray = None, bounding_box: bool = False) -> np.ndarray:
        """Draws regions around the objects in an image.

        Args:
            labels (ndarray): Image containing the objects.
            image (ndarray, optional): Image to draw the regions on. Defaults to None.
            bounding_box (bool, optional): Whether to draw bounding boxes instead of contours. Defaults to False.

        Returns:
            ndarray: Image with regions drawn.

        Todo:
            Use self.colors to use different colors according to the label.

        """
        # If no image is provided, create a blank canvas
        if image is None:
            canvas = np.zeros((labels.shape[0], labels.shape[1], 3)).astype('uint8')
        else:
            # Otherwise, make a copy of the input image
            canvas = image.copy()
        
        # Find the contours of the objects in the image
        contours = cls._findContours(labels)
        
        # Draw bounding boxes or contours around each object
        if bounding_box:
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(canvas, (x,y), (x+w,y+h), (255, 0, 0), 0)
        else:
            cv2.drawContours(canvas, contours, -1, (255, 0, 0), 0)
            
        # Return the canvas with the regions drawn
        return canvas
    
    
    @classmethod
    def __cutVignettes(cls, labels: np.ndarray, image: np.ndarray, bounding_box: bool = False) -> list:
        """Cut out vignettes from an image.
        
        Args:
            labels (ndarray): Image containing the objects.
            image (ndarray): Image to cut out the regions.
            bounding_box (bool, optional): Whether to draw bounding boxes instead of contours. Defaults to False.
        
        Returns:
            pred_cuts: a list of numpy arrays, each containing a vignette.
        
        """
        contours = cls._findContours(labels)
        pred_cuts = []
        if bounding_box:
            # Cut out the vignettes using bounding boxes.
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                pred_cuts.append(image[y:y+h, x:x+w])
        else:
            # Cut out the vignettes using contours.
            for contour_id, contour in enumerate(contours):
                x,y,w,h = cv2.boundingRect(contour)

                # Create a mask of the contour
                mask = np.zeros(image.shape).astype('uint8')
                cv2.fillPoly(mask, [contours[contour_id]], (255, 255, 255))

                # Crop to the box size
                mask_box = mask[y:y+h, x:x+w, :]
                image_box = image.copy()[y:y+h, x:x+w, :]
                
                # Extract the portion of the image corresponding to the contour
                contour_img = cv2.bitwise_and(image_box, mask_box)
                contour_img += cv2.bitwise_not(mask_box)

                pred_cuts.append(contour_img)
        return pred_cuts
    
    
    @classmethod
    def _drawRegions(cls, result: dict, bounding_box: bool = False, verbose: bool = True) -> None:
        """Draw the regions in the image.
        
        Args:
            result (dict): A dictionary containing the image and labels.
            bounding_box (bool): Whether to use bounding box or contours. Default is False.
            verbose (bool): Whether to print verbose output. Default is True.

        """
        result['regions'] = cls.__drawRegions(
            result['labels'],
            result['image'],
            bounding_box
        )
    
    
    @classmethod
    def _cutVignettes(cls, result: dict, bounding_box: bool = False, verbose: bool = True) -> None:
        """Cut out the vignettes from an image and add them to the result dictionary.
        
        Args:
            result (dict): A dictionary containing the image and labels.
            bounding_box (bool): Whether to use bounding box or contours. Default is False.
            verbose (bool): Whether to print verbose output. Default is True.

        """
        result['vignettes'] = cls.__cutVignettes(
            result['labels'],
            result['image'],
            bounding_box
        )
    
    
    def _selectLabels(self, preds: np.ndarray) -> np.ndarray:
        """Select the labels with the highest probability for each pixel.
        
        Args:
            preds (ndarray): numpy array of predicted probabilities for each class.
        
        Returns:
            mask_labels (ndarray): numpy array of the selected labels.
        
        """
        best_preds = np.argmax(preds, axis=0).astype('uint8')
        mask_labels = np.zeros((best_preds.shape[0], best_preds.shape[1], 3)).astype('uint8')
        for i, color in enumerate(self.colors):
            mask_labels[best_preds == i] = color
        return mask_labels
        
        
    def _render(self, result: dict) -> None:
        """Render the predicted labels onto the image.
        
        Args:
            result (dict): dictionary containing the image and predicted probabilities.
        
        """
        # Parse name
        result['name'] = os.path.splitext(os.path.basename(result['path']))[0]

        # Load probas
        result['probasMaps'] = [self.__class__.probasToMaps(result['probas'], c) for c in range(self.num_classes)]
        
        # Convert probas to semantic segmentation
        result['labels'] = self._selectLabels(result['probas'])

        # Load image
        image = cv2.imread(result['path'])
        if self.output_size: # Resize the image so it has self.output_size pixels
            scale_factor = (image.shape[0]*image.shape[1] / self.output_size)**0.5
            new_width = int(image.shape[1] // scale_factor)
            new_height = int(image.shape[0] // scale_factor)
            image = cv2.resize(image, (new_width, new_height))
        result['image'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
    def _saveResults(self) -> None:
        """Saves inference results as image files in the output directory."""
        for id_page, result in enumerate(self.results):
            for id_label, vignette in enumerate(result['vignettes']):
                file_path = os.path.join(self.output_dir, f"{id_page}.{id_label}.png")
                cv2.imwrite(file_path, vignette)
    
    
    def _dataCSVToFolder(self, csv_path: str, empty_folder: bool = True) -> None:
        """Copies the test data from the CSV file to the data directory.
        
        Args:
            csv_path (str): The path to the CSV file containing the test data.
            empty_folder (bool): Whether to remove existing data in the data directory.
        
        """
        if empty_folder: # Remove existing data
            for file in glob.glob(os.path.join(self.data_dir, '*.*')):
                os.remove(file)
        
        data_path = os.path.dirname(csv_path)
        test_data = pd.read_csv(csv_path, header= None)
        if test_data.shape[1] == 2:
            for id, row in test_data.iterrows():
                src_file = os.path.join(data_path, row[0])
                shutil.copy(src_file, os.path.join(self.data_dir))
        
        
    def _setupInference(self, batch_size= 4, output_size= None) -> dict:
        """Sets up the parameters for inference.
        
        Args:
            batch_size (int): The batch size for inference.
            output_size (tuple): The size to resize the input image before inference.
        
        Returns:
            dict: The inference parameters.
        
        """
        if output_size:
            transforms = [{"type": "fixed_size_resize", "output_size": output_size}]
        else:
            transforms = []
        
        dataset_params = {
            "type": "folder",
            "folder": self.data_dir,
            "pre_processing": {"transforms": transforms}
        }
        
        model_state_dict = sorted(glob.glob(os.path.join(self.model_dir, 'best_model_checkpoint_miou=*.pth')))
        
        if not model_state_dict:
            no_model_error = Exception("No model! Cannot perform inference")
            _LOGGER.error(no_model_error)
            raise no_model_error
        
        model_params = {
            "model": {
                    "encoder": "resnet50",
                    "decoder": {"decoder_channels": [512, 256, 128, 64, 32], "max_channels": 512}
                },
                "num_classes": self.num_classes,
                "model_state_dict": model_state_dict[-1],
                "device": "cuda:0"
        }

        process_params = {
            'data': dataset_params,
            'model': model_params,
            'batch_size': batch_size,
            'add_path': True
        }
        return process_params
    
    
    def start(self, batch_size= 4, drawRegions= True, cutVignettes= True, bounding_box= False, verbose= True) -> list:
        """Starts the inference process and saves the vignettes in the output directory.
        
        Args:
            batch_size (int): The batch size for inference.
            drawRegions (bool): Whether to draw bounding boxes on the input images.
            cutVignettes (bool): Whether to cut out the predicted regions from the input images.
            bounding_box (bool): Whether to include the bounding box coordinates in the output.
            verbose (bool): Whether to print progress messages during the inference process.
        
        Returns:
            list: The inference results. They contain images with drawn regions, vignettes, probability maps, paths and names.
        
        """
        dhPredictor = PredictProcess.from_params(
            Params(
                self._setupInference(batch_size)
            )
        )
        _LOGGER.info(f"Starting inference of model from {self.model_dir}")
        self.results = dhPredictor.process()
        
        _LOGGER.info("Post-process predictions")
        for result in tqdm(self.results, desc= _DESC_PROGRESSBAR_POSTPROCESS, disable= not verbose):
            self._render(result)
            if drawRegions:
                self.__class__._drawRegions(result, bounding_box, verbose)
            if cutVignettes:
                self.__class__._cutVignettes(result, bounding_box, verbose)
        self._saveResults()
        _LOGGER.info(f"Results can be found in {self.output_dir}")
        return self.results
