import numpy as np
import pandas as pd
from PIL import Image
import os, glob, cv2

from dh_segment_torch.config import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.training import Trainer as dhTrainer
from dh_segment_torch.inference import PredictProcess


class ModelUser:
    def __init__(self, labels):
        self.results_dir = labels
        self.data_dir = os.path.join(results_dir, 'data')
        self.model_dir = os.path.join(results_dir, "model")

        if not os.path.exists(self.data_dir):
            raise Exception(f"No dataset at {self.data_dir}. Please use data_preparation module before training.")
        os.makedirs(self.results_dir, exist_ok= True)
    
    
    @property
    def results_dir(self):
        return self.results_dir
    
    
    @results_dir.setter
    def results_dir(self, labels):
        #labels = labels.sort() # in the case of multilabels
        labels_str = '_'.join(labels)
        self.results_dir = os.path.join("results", labels_str)

        
        
class Trainer(ModelUser):
    def __init__(self, labels):
        super().__init__(labels)
        self.tensorboard_dir = os.path.join(self.results_dir, 'tensorboard', 'log')
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
       
    
    def _setupTraining(self, batch_size= 4, epochs= 100, learning_rate: 1e-4, gamma_exp_lr= 0.9995, evaluate_every_epoch= 5, val_patience= 4, output_size= 1e6, repeat_dataset= 4):
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
        
        
    def train(self):
        params = self._setupTraining()
        trainer = dhTrainer.from_params(params)
        trainer.train()

        
        
class Predictor(ModelUser):
    def __init__(self, labels):
        super().__init__(labels)
        self.output_dir = os.path.join(self.results_dir, 'predictions')
        self.num_classes = 0
        self.results = None
        
        
    def _setupInference(self):
        dataset_params = {
            "type": "folder",
            "folder": os.path.join(self.data_dir, 'images'),
            "pre_processing": {"transforms": [{"type": "fixed_size_resize", "output_size": 1e6}]}
        }

        model_params = {
            "model": {
                    "encoder": "resnet50",
                    "decoder": {"decoder_channels": [512, 256, 128, 64, 32], "max_channels": 512}
                },
                "num_classes": 2,
                "model_state_dict": sorted(glob.glob(os.path.join(self.model_dir, 'best_model_checkpoint_miou=*.pth')))[-1],
                "device": "cuda:0"
        }

        process_params = {
            'data': dataset_params,
            'model': model_params,
            'batch_size': 4,
            'add_path': True
        }
        return process_params
    
    
    def inference(self):
        dhPredictor = PredictProcess.from_params(
            Params(
                self._setupInference()
            )
        )
        self.results = dhPredictor.process()
        self._convertPredictions() 
        
        
    def _convertPredictions(self):
        for result in self.results:
            # Set number of classes
            num_classes_result = result['probas'].shape[0]
            if num_classes_result > self.num_classes:
                self.num_classes = num_classes_result
                
            # Convert probas to semantic segmentation
            result['semantic_segment'] = postprocess(result.pop(probas))
            
            # Load image
            result['image'] = cv2.cvtColor(
                cv2.imread(result['path']),
                cv2.COLOR_BGR2RGB
            )
            
            # Parse name
            result['name'] = os.path.splitext(os.path.basename(result.pop('path')))[0]
    
    
    @staticmethod
    def postprocess(preds):
        best_preds = np.argmax(preds, axis=0).astype('uint8')
        canvas = np.zeros((best_preds.shape[0], best_preds.shape[1], 3)).astype('uint8')
        for i, color in enumerate(colors):
            canvas[best_preds == i] = color
        return canvas
    
    
    @staticmethod
    def _findContours(preds):
        # Threshold the image to create a binary image
        grey_preds = cv2.cvtColor(preds, cv2.COLOR_BGR2GRAY)
        _, thresh_preds = cv2.threshold(grey_preds, 1, 255, cv2.THRESH_BINARY)

        # Loop over the contours and draw bounding boxes around them
        contours, _ = cv2._findContours(thresh_preds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


    def drawRegions(preds, image= None, bounding_box= False):
        if image is None:
            canvas = np.zeros((preds.shape[0], preds.shape[1], 3)).astype('uint8')
        else:
            canvas = image.copy()
        contours = _findContours(preds)
        if bounding_box:
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(canvas, (x,y), (x+w,y+h), 255, 0)
        else:
            cv2.drawContours(canvas, contours, -1, 255, 0)
        return canvas
    
    def cutVignettes(preds, image, bounding_box= False):
        contours = findContours(preds)
        pred_cuts = []
        if bounding_box:
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                contour_img = np.zeros((h, w), dtype=np.uint8)
                translated_contour = contour - np.array([x, y])
                cv2.drawContours(contour_img, [translated_contour], 0, 255, 0)
                pred_cuts.append(contour_img)
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
