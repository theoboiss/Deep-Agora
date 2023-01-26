import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import os, glob, shutil

from dh_segment_torch.data import ColorLabels

from training_data.xml_parser import *


desc_progressbar_images =   "Building label masks"
desc_progressbar_masks =    "Copying images      "


class DataStructure(object):
    def __init__(self, dir_data, dir_images, dir_labels= None, dir_annotations= None):
        self.wrapDirs(
            dir_data,
            dir_images,
            dir_labels,
            dir_annotations
        )

    def wrapDirs(self, dir_data, dir_images, dir_labels, dir_annotations, child_dir_data= ""):
        self.dir_data = os.path.join(dir_data, child_dir_data) if child_dir_data else dir_data
        self.dir_images = os.path.join(dir_data, dir_images)
        self.dir_labels = os.path.join(dir_data, dir_labels) if dir_labels else None
        self.dir_annotations = os.path.join(dir_data, dir_annotations) if dir_annotations else None
    
    def wrapDirsSelf(self, wrapper_dir_data):
        self.wrapDirs(
            dir_data= wrapper_dir_data,
            child_dir_data= self.dir_data,
            dir_images= self.dir_images,
            dir_labels= self.dir_labels,
            dir_annotations= self.dir_annotations
        )
    
    @staticmethod
    def collectPaths(dir, extension):
        """
        Extract the absolute path of all the annotation files.
        """
        paths = glob.glob(os.path.join(dir, extension))
        assert len(paths)
        return sorted(paths)


class AnnotationEncoder:
    def __init__(self, dir_annotations):
        self.files = DataStructure.collectPaths(dir_annotations, '*.xml')
        self.valid_tags = collectAllTags(self.files)
        self.codes_labels = None
    
    def _chooseLabels_(self):
        """
        Return chosen names of labels among the valid tags of the annotations.
        WARNING: names of labels must be entered in a hierarchical order.
        """
        print("Enter your labels in a hierarchical order separated by a space ' ':")
        for tag in self.valid_tags: print(tag)
        print()

        # Input targeted labels
        valid = False
        while not valid:
            chosen_labels = input("> ").replace('\'','') \
                                       .replace(',','') \
                                       .split(' ')
            valid = True
            for label in chosen_labels:
                valid &= (label in self.valid_tags)
                if not valid:
                    print("Parsing did not work. Please just copy-paste from the list above.")
                    break
        print()
        return chosen_labels

    def _getPageSize_(self, page):
        width = int(extractAttributeElem(page, 'imageWidth'))
        height = int(extractAttributeElem(page, 'imageHeight'))
        return width, height

    def encodeLabels(self, names_labels) -> dict:
        """
        Update the dictionnary that encodes each combination of label with a color and a one-hot encoding.
        """
        if 'Background' not in names_labels:
            names_labels[0:0] = ['Background']
        # TODO Adapt for multilabel
        encoding_labels = ColorLabels.from_labels_multilabel(names_labels)
        codes_labels = dict()
        for index, label in enumerate(encoding_labels.labels): # TODO Add combinations
            codes_labels[label] = {'color': encoding_labels.colors[index],
                                    'onehot': encoding_labels.one_hot_encoding[index]}
        self.codes_labels = codes_labels
        return encoding_labels

    def encodeLabelsInteractive(self) -> dict:
        """
        Update the dictionnary that encodes each combination of label chosen by the user with a color and a one-hot encoding.
        WARNING: names of labels must be entered in a hierarchical order.
        """
        names_labels = self._chooseLabels_()
        encoding_labels = self.encodeLabels(names_labels)
        return encoding_labels, names_labels

    def extractCoordsLabels(self) -> dict:
        shapes_label_file = dict()
        for path in self.files:
            name_file = os.path.basename(path)[:-4]
            page = extractTagElem(path, self.valid_tags['Page'])
            shapes_label_file[name_file] = {
                'size': self._getPageSize_(page),
                'coords': None
            }
            shapes_label_file[name_file]['coords'] = extractAttributesTag(
                'Coords',
                'points',
                self.codes_labels.keys(),
                self.valid_tags,
                page
            )
        return shapes_label_file

    @staticmethod
    def debugCoordinates(shapes_label_file):
        """
        Debug found coordinates.
        """
        print("Found coordinates for:")
        for path, coords_label in shapes_label_file.items():
            print(path)
            for label, coords in coords_label['coords'].items():
                if label == 'Background':
                    width, height = coords_label['size']
                    print(f"({width}x{height})")
                else:
                    print('\t', label, coords)
                    print('\t', f"({len(coords)} object{'s' if len(coords)-1 else ''})")


    @staticmethod
    def debugCodesLabels(codes_labels):
        """
        Debug label encodings.
        """
        print("Encoding of labels:")
        for label, codes in codes_labels.items():
            print(label)
            print(f"\t Color: {codes['color']}")
            print(f"\t One-hot encoding: {codes['onehot']}")
        print()


class MaskBuilder:
    def __init__(self, dir_masks, codes_labels, shapes_label_file):
        self.dir_masks = dir_masks
        self.codes_labels = codes_labels
        self.shapes_label_file = shapes_label_file
        self.masks = []

    def _buildMask_(self, coords_label, width_image, height_image, name):
        canvas = Image.fromarray(np.ones((height_image, width_image, 3), dtype='uint8'))
        draw = ImageDraw.Draw(canvas)
        for label, coords in coords_label.items():
            for object in coords:
                MaskBuilder.drawObject(object, self.codes_labels[label]['color'], draw)
        name += ".png"
        path_mask = os.path.join(self.dir_masks, name)
        self.masks.append(path_mask)
        canvas.save(path_mask, format='PNG')

    @staticmethod
    def drawObject(points, colour, draw):
        points = [tuple(int(xy) for xy in point.split(',')) for point in points.split(' ')]
        draw.polygon(points, fill= colour)
    
    def buildAllMasks(self):
        for name, shapes_label in tqdm(self.shapes_label_file.items(), desc= desc_progressbar_images):
            width, height = shapes_label['size']
            self._buildMask_(shapes_label['coords'], width, height, name)
        return self.masks


class DataPatcher:
    def __init__(self, original_data: DataStructure, new_data: DataStructure):
        """
        Init directories of data to patch.
        """
        if  not original_data.dir_annotations \
        and     original_data.dir_labels \
        and not new_data.dir_labels:
            raise ValueError("The original data and the new data do not have compatible structure specifications.")

        self.original_data = original_data
        self.new_data = new_data
        self._ensureOriginalDataDirs_()

    def _ensureOriginalDataDirs_(self):
        for dir in vars(self.original_data).values():
            if dir:
                assert os.path.isdir(dir)

    def _ensureNewDataDirs_(self):
        for dir in vars(self.new_data).values():
            if dir:
                os.makedirs(dir, exist_ok= True)

    def _copyImages_(self, extensions= ('*.jpg', '*.tif', '*.png')):
        """
        Copy images from the original image folder to the new one.
        """
        images_extensions = [
            (
                glob.glob(
                    os.path.join(self.original_data.dir_images, os.path.join("**", e)),
                    recursive= True
                ),
                e
            )
            for e in extensions
        ]
        images_unique_extension = [(i, e) for i, e in images_extensions if len(i)]
        assert len(images_unique_extension) == 1
        images_extension = images_unique_extension[0]

        for image in tqdm(images_extension[0], desc= desc_progressbar_masks):
            shutil.copy(image, self.new_data.dir_images)
        return DataStructure.collectPaths(self.new_data.dir_images, images_extension[1])

    @staticmethod
    def colorLabelsToTXT(codes_labels, path):
        class_data = {'label' : [], 'color_rgb' : [], 'parts' : [], 'actions' : []}
        for label, codes in codes_labels.items():
            class_data["label"].append(
                label
            ),
            class_data["color_rgb"].append(
                ','.join([str(codes['color'][x]) for x in range(3)])
            ),
            class_data["parts"].append(
                ','.join([str(codes['onehot'][x]) for x in range(len(codes['onehot']))])
            ),
            class_data["actions"].append(
                ''
            )
        assert len(class_data.values())
        with open(path, "w", encoding="utf-8") as classfile_txt:
            classfile_txt.write(f"# {':'.join(class_data.keys())}\n")
            lines = zip(*class_data.values())
            for line in lines:
                classfile_txt.write(':'.join(line)+'\n')

    def patch(self, names_labels= None, debug= False):
        # Compute encoding of the labels
        ae = AnnotationEncoder(self.original_data.dir_annotations)
        if names_labels:
            encoding_labels = ae.encodeLabels(names_labels)
        else:
            encoding_labels, names_labels = ae.encodeLabelsInteractive()

        # Extract the coordinates of the labels
        shapes_label_file = ae.extractCoordsLabels()
        
        # Debug annotation extractions
        if debug:
            AnnotationEncoder.debugCodesLabels(ae.codes_labels)
            AnnotationEncoder.debugCoordinates(shapes_label_file)

        # Group data by selected label
        dir_category = '_'.join([
            label for label in names_labels
            if label != 'Background'
        ])
        self.new_data.wrapDirsSelf(dir_category)
        self._ensureNewDataDirs_()

        # Copy images
        images = self._copyImages_()

        # Build label masks
        mb = MaskBuilder(self.new_data.dir_labels,
                         ae.codes_labels,
                         shapes_label_file)
        masks = mb.buildAllMasks()

        # Write data summary
        df = pd.DataFrame(data= zip(images, masks))
        df.to_csv(os.path.join(self.new_data.dir_data, "data.csv"), header=False)

        # Write encoding of the labels in the class file
        DataPatcher.colorLabelsToTXT(ae.codes_labels, os.path.join(self.new_data.dir_data, "classfile.txt"))
