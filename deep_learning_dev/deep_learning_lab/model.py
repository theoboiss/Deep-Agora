import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os, glob, cv2, shutil

from dh_segment_torch.config import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.training import Trainer as dhTrainer
from dh_segment_torch.inference import PredictProcess

from deep_learning_lab import logging


_LOGGER = logging.getLogger(__name__)

_DESC_PROGRESSBAR_POSTPROCESS = "Post-processing predictions "
_DESC_PROGRESSBAR_REGIONS =     "Drawing regions on images   "
_DESC_PROGRESSBAR_VIGNETTES =   "Extracting vignettes        "

        

def _n_colors(n):
    """
    Returns a list of n random RGB colors generated with a constant seed.
    """
    assert n
    colors = [(0, 0, 0)]
    rng = np.random.default_rng(0);
    r = int(rng.random() * 256)
    g = int(rng.random() * 256)
    b = int(rng.random() * 256)
    step = 256 / n
    for _ in range(n-1):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        colors.append((r, g, b))
    return colors



class ModelUser:
    def __init__(self, labels, input_dir, workdir= "results"):
        if workdir:
            self.workdir = os.path.join(workdir, '_'.join(labels))
            os.makedirs(self.workdir, exist_ok= True)
        self.model_dir = os.path.join(self.workdir, "model")
        self.data_dir = os.path.join(self.workdir, input_dir)
        os.makedirs(self.data_dir, exist_ok= True)
        
        
    def __str__(self):
        return self.__class__.__name__ + ": " + self.__dict__

        
        
class Trainer(ModelUser):
    def __init__(self, labels, input_dir= "training_data", workdir= "results"):
        super().__init__(labels, input_dir, workdir)
        
        self.tensorboard_dir = os.path.join(self.workdir, 'tensorboard', 'log')
        self._setupEnvironment()

        
    def _setupEnvironment(self):
        params = {
            'data_path' : self.data_dir, # Path to write the data
            'data_splitter': {'train_ratio': 0.80, 'val_ratio': 0.10, 'test_ratio': 0.10}, # splitting ratio of the data
        }
        
        
        ## Process parameters

        relative_path = params.pop("relative_path", True)

        params.setdefault("labels_dir", os.path.join(self.data_dir, "labels"))
        labels_dir = params.get("labels_dir")

        params.setdefault("images_dir", os.path.join(self.data_dir, "images"))
        images_dir = params.get("images_dir")

        params.setdefault("csv_path", os.path.join(self.data_dir, "data.csv"))

        data_splitter_params = params.pop("data_splitter", None)
        train_csv_path = params.pop("train_csv", os.path.join(self.data_dir, "train.csv"))
        val_csv_path = params.pop("val_csv", os.path.join(self.data_dir, "val.csv"))
        test_csv_path = params.pop("test_csv", os.path.join(self.data_dir, "test.csv"))

        
        ## List labels and images in CSV file

        labels_list = sorted(glob.glob(os.path.join(labels_dir, '*.*')))
        images_list = sorted(glob.glob(os.path.join(images_dir, '*.*')))

        data = pd.DataFrame({'image': images_list, 'label': labels_list})
        data.to_csv(params['csv_path'], header=False, index=False)

        if relative_path:
            data['image'] = data['image'].apply(lambda path: os.path.join("images", os.path.basename(path)))
            data['label'] = data['label'].apply(lambda path: os.path.join("labels", os.path.basename(path)))

        ## Divide data set in train, validation and test sets
        
        if data_splitter_params:
            data_splitter = DataSplitter.from_params(data_splitter_params)
            data_splitter.split_data(data, train_csv_path, val_csv_path, test_csv_path)
            
        return params
       
    
    def _setupTraining(self, batch_size, epochs, learning_rate, gamma_exp_lr, evaluate_every_epoch, val_patience, output_size, repeat_dataset):
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
        
        
    def train(self, batch_size= 1, epochs= 1, learning_rate= 1e-1, gamma_exp_lr= 1.0, evaluate_every_epoch= 1, val_patience= 1, repeat_dataset= 1, output_size= 1e6):
        params = self._setupTraining(batch_size, epochs, learning_rate, gamma_exp_lr, evaluate_every_epoch, val_patience, output_size, repeat_dataset)
        trainer = dhTrainer.from_params(params)
        trainer.train()
        _LOGGER.info(f"Model trained and serialized in {model_dir}")
        
        
        
class Predictor(ModelUser):
    def __init__(self, labels, input_dir= 'inference_data', output_dir= None, output_size= None, from_csv= None, reset_input= True):
        super().__init__(labels, input_dir)
        
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(self.workdir, 'predictions')
            
        self.num_classes = len(labels)+1
        self.colors = _n_colors(self.num_classes)
        self.output_size = output_size
        self.results = None
        
        if from_csv:
            self._dataCSVToFolder(os.path.join(self.workdir, from_csv), reset_input)
            
        if not os.path.exists(self.output_dir):
            _LOGGER.info(f"{self.output_dir} created")
        os.makedirs(self.output_dir, exist_ok= True)
        
        
    @staticmethod
    def probasToMaps(probas, id_class):
        maps_probas_class = probas[id_class]
        map_probas = np.around(maps_probas_class*255)
        return map_probas.astype('uint8')
    
    
    @staticmethod
    def _findContours(labels):
        # Threshold the image to create a binary image
        grey_labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
        _, thresh_labels = cv2.threshold(grey_labels, 1, 255, cv2.THRESH_BINARY)

        # Loop over the contours and draw bounding boxes around them
        contours, _ = cv2.findContours(thresh_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    
    @classmethod
    def __drawRegions(cls, labels, image= None, bounding_box= False):
        if image is None:
            canvas = np.zeros((labels.shape[0], labels.shape[1], 3)).astype('uint8')
        else:
            canvas = image.copy()
        contours = cls._findContours(labels)
        if bounding_box:
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(canvas, (x,y), (x+w,y+h), (255, 0, 0), 0)
        else:
            cv2.drawContours(canvas, contours, -1, (255, 0, 0), 0)
        return canvas
    
    
    @classmethod
    def __cutVignettes(cls, labels, image, bounding_box= False):
        contours = cls._findContours(labels)
        pred_cuts = []
        if bounding_box:
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                pred_cuts.append(image[y:y+h, x:x+w])
        else:
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
    def _drawRegions(cls, result, bounding_box= False, verbose= True):
        result['regions'] = cls.__drawRegions(
            result['labels'],
            result['image'],
            bounding_box
        )
    
    
    @classmethod
    def _cutVignettes(cls, result, bounding_box= False, verbose= True):
        result['vignettes'] = cls.__cutVignettes(
            result['labels'],
            result['image'],
            bounding_box
        )
    
    
    def _selectLabels(self, preds):
        best_preds = np.argmax(preds, axis=0).astype('uint8')
        mask_labels = np.zeros((best_preds.shape[0], best_preds.shape[1], 3)).astype('uint8')
        for i, color in enumerate(self.colors):
            mask_labels[best_preds == i] = color
        return mask_labels
        
        
    def _render(self, result):
        # Parse name
        result['name'] = os.path.splitext(os.path.basename(result['image_path']))[0]

        # Load probas
        probas = np.load(result['probas_path'])
        result['probasMaps'] = [self.__class__.probasToMaps(probas, c) for c in range(self.num_classes)]
        
        # Convert probas to semantic segmentation
        result['labels'] = self._selectLabels(probas)

        # Load image
        image = cv2.imread(result['image_path'])
        if self.output_size: # Resize the image so it has self.output_size pixels
            scale_factor = (image.shape[0]*image.shape[1] / self.output_size)**0.5
            new_width = int(image.shape[1] // scale_factor)
            new_height = int(image.shape[0] // scale_factor)
            image = cv2.resize(image, (new_width, new_height))
        result['image'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    def _dataCSVToFolder(self, csv_path, empty_folder= True):
        if empty_folder: # Remove existing data
            for file in glob.glob(os.path.join(self.data_dir, '*.*')):
                os.remove(file)
        
        data_path = os.path.dirname(csv_path)
        test_data = pd.read_csv(csv_path, header= None)
        if test_data.shape[1] == 2:
            for id, row in test_data.iterrows():
                src_file = os.path.join(data_path, row[0])
                shutil.copy(src_file, os.path.join(self.data_dir))
        
        
    def _setupInference(self, batch_size= 4, output_size= None):
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
            msg = "No model! Cannot perform inference"
            _LOGGER.error(msg)
            raise Exception(msg)
        
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
            
            
    def load(self, verbose= True):
        self.results = []
        file_dict = {}
        
        probas_paths = glob.glob(os.path.join(self.output_dir, '*.npy'))
        image_paths = glob.glob(os.path.join(self.data_dir, '*.*'))

        # Match pairs between probas and images
        for path in probas_paths + image_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in file_dict:
                file_dict[name] = {'probas_path': None, 'image_path': None}
            if path in probas_paths:
                file_dict[name]['probas_path'] = path
            else:
                file_dict[name]['image_path'] = path

        for name, files in file_dict.items():
            if files['probas_path'] and files['image_path']:
                self.results.append({
                    'probas_path': files['probas_path'],
                    'image_path': files['image_path'],
                    'name': name
                })
            elif files['image_path']:
                msg = f"{name} is missing a probability file! Apply the start method to segment it"
                print(msg)
                _LOGGER.error(msg)
            elif files['probas_path']:
                _LOGGER.info(f"{name} is missing an image file")
    
    
    def start(self, batch_size= 4, save_probas= True):
        dhPredictor = PredictProcess.from_params(
            Params(
                self._setupInference(batch_size)
            )
        )
        _LOGGER.info("Start inference")
        if save_probas: # Save predictions
            for file in glob.glob(os.path.join(self.output_dir, '*.npy')):
                os.remove(file)
            dhPredictor.process_to_probas_files(self.output_dir)
            _LOGGER.info(f"Predictions saved in {self.output_dir}")
            self.load()
        else: # Do not save predictions
            self.results = dhPredictor.process()
        
        
    def postProcess(self, drawRegions= True, cutVignettes= True, bounding_box= False, verbose= True):
        _LOGGER.info("Post-process predictions")
        for result in tqdm(self.results, desc= _DESC_PROGRESSBAR_POSTPROCESS, disable= not verbose):
            self._render(result)
            if drawRegions:
                self.__class__._drawRegions(result, bounding_box, verbose)
            if cutVignettes:
                self.__class__._cutVignettes(result, bounding_box, verbose)
        return self.results
