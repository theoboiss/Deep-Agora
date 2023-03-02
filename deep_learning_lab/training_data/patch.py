"""
=======================================
Data set patching for dhSegment
=======================================

This module offers tools to build training data under the right
format for dhSegment models.

It converts a dataset with serialized annotations into a dataset
that meets the requirements of dhSegment.

Its interactive functionnality allows the user to design diverse training
datasets according to the elements of content they want to extract.
For the moment, it can only patch datasets under the PAGE format.

"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import os, glob

from dh_segment_torch.data import ColorLabels, all_one_hot_and_colors, get_all_one_hots

from training_data.xml_parser import *


TEMP_ROOT_DIR = "" # "" if deactivated. Useful for Google Collab
DESC_PROGRESSBAR_IMAGES = "Building label masks "
DESC_PROGRESSBAR_MASKS =  "Copying images       "



def resizeImgAndSave(img: Image, path, extension, width= None, height= None):
    """
    Resize the PIL.Image and save it to path with the specified extension.
    """
    # Resize the image
    if width or height:
        if height and not width:
            width = int((height / float(img.size[1])) * float(img.size[0]))
        elif not height:
            height = int((width / float(img.size[0])) * float(img.size[1]))
        img = img.resize((width, height), Image.ANTIALIAS)

    # Then copy it under JPEG format
    img.save(path, format= extension)



class DataStructure:
    """
    Description of a dataset in the file system.
    The dataset can be patched or not.
    
    All the dir attributes are supposed to be subdirectories of the dir_data directory.
    """
    def __init__(self, dir_data, dir_images, dir_labels= None, dir_annotations= None):
        self.wrapDirs(
            dir_data,
            dir_images,
            dir_labels,
            dir_annotations
        )
    
    @staticmethod
    def collectPaths(dir, extension):
        """
        Extract the absolute path of all the files.
        """
        paths = glob.glob(os.path.join(dir, extension))
        assert len(paths)
        return sorted(paths)

    def wrapDirs(self, dir_data, dir_images, dir_labels, dir_annotations, child_dir_data= ""):
        """
        Setter of the attributes that wraps all the directories into the dir_data directory.
        
        If child_dir_data is defined, it is inserted at the end of the path of dir_data.
        """
        self.dir_data = os.path.join(dir_data, child_dir_data) if child_dir_data else dir_data
        self.dir_images = os.path.join(dir_data, dir_images)
        self.dir_labels = os.path.join(dir_data, dir_labels) if dir_labels else None
        self.dir_annotations = os.path.join(dir_data, dir_annotations) if dir_annotations else None
    
    def wrapDirsSelf(self, wrapper_dir_data):
        """
        Setter of the attributes that wraps all the directories into wrapper_dir_data.
        
        If TEMP_ROOT_DIR is defined, it is the wrapper of wrapper_dir_data and it is removed from previous insertions.
        """
        if TEMP_ROOT_DIR:
            wrapper_dir_data = os.path.join(
                TEMP_ROOT_DIR,
                wrapper_dir_data
            )
            self.dir_data = self.dir_data[self.dir_data.find(TEMP_ROOT_DIR)+len(TEMP_ROOT_DIR):]
            self.dir_images = self.dir_images[self.dir_images.find(TEMP_ROOT_DIR)+len(TEMP_ROOT_DIR):]
            self.dir_labels = self.dir_labels[self.dir_labels.find(TEMP_ROOT_DIR)+len(TEMP_ROOT_DIR):] if self.dir_labels else None
            self.dir_annotations = self.dir_annotations[self.dir_annotations.find(TEMP_ROOT_DIR)+len(TEMP_ROOT_DIR):] if self.dir_annotations else None
        
        self.wrapDirs(
            dir_data= wrapper_dir_data,
            child_dir_data= self.dir_data,
            dir_images= self.dir_images,
            dir_labels= self.dir_labels,
            dir_annotations= self.dir_annotations
        )
    
    def __str__(self):
        return ", ".join((
            "DataStructure: "+
            "dir_data= "+self.dir_data,
            "dir_images= "+self.dir_images,
            "dir_labels= "+str(self.dir_labels),
            "dir_annotations= "+str(self.dir_annotations)
        ))



def n_colors(n):
    """
    Returns a list of n random RGB colors generated with a constant seed.
    """
    colors = []
    rng = np.random.default_rng(0);
    r = int(rng.random() * 256)
    g = int(rng.random() * 256)
    b = int(rng.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        colors.append((r, g, b))
    return colors

class constantColorLabels(ColorLabels):
    """
    Overloaded class of ColorLabels from dhSegment that always returns the same colours.
    """
    
    @classmethod
    def from_labels(cls, labels):
        """
        Returns an instance of ColorLabels of atomic labels.
        """
        num_classes = len(labels)
        colors = n_colors(num_classes)
        return cls(colors, labels=labels)
    
    @classmethod
    def from_labels_multilabel(cls, labels):
        """
        Returns an instance of ColorLabels of multi-labels.
        """
        num_classes = len(labels)

        num_tries_left = 10
        while num_tries_left:
            num_tries_left -= 1
            base_colors = n_colors(num_classes)
            one_hot_encoding, colors = all_one_hot_and_colors(base_colors)
            if len(colors) == len(set(colors)):
                break
        else:
            print(
                f"WARNING: Could not find a color combinatation for {num_classes}. "
                "Falling back on one color per one hot encoding."
            )
            one_hot_encoding = get_all_one_hots(num_classes).tolist()
            colors = n_colors(len(one_hot_encoding))
        return cls(colors, one_hot_encoding, labels)



class AnnotationEncoder:
    """
    
    """
    def __init__(self, dir_annotations):
        self.files = DataStructure.collectPaths(dir_annotations, '*.xml')
        self.tag_maps = collectAllTags(self.files, warning= False)
        self.codes_labels = None

    def _getPageSize_(self, page):
        try:
            width = int(extractAttributeElem(page, 'imageWidth'))
            height = int(extractAttributeElem(page, 'imageHeight'))
        except Exception:
            raise Exception(f"WARNING: The Page attribute has no size value.")
        return width, height
    
    def chooseLabels(self):
        """
        Return chosen names of labels among the valid tags of the annotations.
        WARNING: names of labels must be entered in a hierarchical order.
        """
        print("Enter your labels in a hierarchical order separated by a space ' ':")
        for tag in self.tag_maps: print(tag)
        print()

        # Input targeted labels
        valid = False
        while not valid:
            chosen_labels = input("> ").replace('\'','') \
                                       .replace(',','') \
                                       .split(' ')
            valid = True
            for label in chosen_labels:
                valid &= (label in self.tag_maps)
                if not valid:
                    print("Parsing did not work. Please just copy-paste from the list above.")
                    break
        print()
        return chosen_labels

    def encodeLabels(self, names_labels, add_background= True) -> dict:
        """
        Update the dictionnary that encodes each combination of label with a color and a one-hot encoding.
        """
        if add_background and 'Background' not in names_labels:
            names_labels = ['Background'] + names_labels
        #encoding_labels = constantColorLabels.from_labels_multilabel(names_labels)
        encoding_labels = constantColorLabels.from_labels(names_labels)
        codes_labels = dict()
        for index, label in enumerate(encoding_labels.labels): # TODO Add multilabels
            codes_labels[label] = {'color': encoding_labels.colors[index]}
                                   #'onehot': encoding_labels.one_hot_encoding[index]}
        self.codes_labels = codes_labels
        return encoding_labels

    def encodeLabelsInteractive(self) -> dict:
        """
        Update the dictionnary that encodes each combination of label chosen by the user with a color and a one-hot encoding.
        WARNING: names of labels must be entered in a hierarchical order.
        """
        names_labels = self.chooseLabels()
        encoding_labels = self.encodeLabels(names_labels)
        return encoding_labels, names_labels

    def extractCoordsLabels(self, debug= False) -> dict:
        shapes_label_file = dict()
        anomalies = set()
        for path in self.files:
            name_file = os.path.basename(path)[:-4]
            try:
                page = extractTagElem(path, 'Page', self.tag_maps['Page'])
                shapes_label_file[name_file] = {
                    'size': self._getPageSize_(page),
                    'coords': extractAttributesTag(
                        'Coords',
                        'points',
                        self.codes_labels.keys(),
                        self.tag_maps,
                        page
                    )
                }
            except Exception as e:
                anomalies.add(name_file)
                if debug:
                    print(e)
        return shapes_label_file, anomalies

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
            #print(f"\t One-hot encoding: {codes['onehot']}")
        print()



class MaskBuilder:
    def __init__(self, dir_masks, codes_labels, shapes_label_file):
        self.dir_masks = dir_masks
        self.codes_labels = codes_labels
        self.shapes_label_file = shapes_label_file
        self.masks = []

    def _buildMask_(self, coords_label, original_size, new_size, name):
        canvas = Image.fromarray(np.ones((original_size[1], original_size[0], 3), dtype='uint8'))
        draw = ImageDraw.Draw(canvas)
        for label, coords in coords_label.items():
            for object in coords:
                self.__class__.drawObject(object, self.codes_labels[label]['color'], draw)
        name += ".png"
        path_mask = os.path.join(self.dir_masks, name)
        self.masks.append(path_mask)
        resizeImgAndSave(canvas, path_mask, 'PNG', new_size[0], new_size[1])

    @staticmethod
    def drawObject(points, colour, draw):
        points = [tuple(int(xy) for xy in point.split(',')) for point in points.split(' ')]
        draw.polygon(points, fill= colour)

    def buildAllMasks(self, new_size, verbose= True):
        for name, shapes_label in tqdm(self.shapes_label_file.items(), desc= DESC_PROGRESSBAR_IMAGES, disable= not verbose):
            self._buildMask_(shapes_label['coords'], shapes_label['size'], new_size, name)
        return self.masks



class DataPatcher:
    def __init__(self, original_data: DataStructure, new_data: DataStructure = None):
        """
        Init directories of data to patch.
        """
        if  original_data.dir_annotations   is None     \
        and new_data                        is not None \
        and new_data.dir_labels             is not None :
            raise ValueError("The original data and the new data do not have compatible structure.")

        self.original_data = original_data
        self.new_data = new_data
        self._ensureOriginalDataDirs_()

    def _ensureOriginalDataDirs_(self):
        for dir in vars(self.original_data).values():
            if dir and not os.path.isdir(dir):
                raise Exception(f"{dir} does not exist.")

    def _ensureNewDataDirs_(self):
        for dir in vars(self.new_data).values():
            if dir:
                os.makedirs(dir, exist_ok= True)

    def _copyImages_(self, new_size= (None, None), anomalies= set(), extensions= ['*.jpg', '*.tif', '*.png'], verbose= True):
        """
        Copy images from the original image folder to the new one.
        """
        images = set()
        images_to_remove = set()
        extensions += [ext.upper() for ext in extensions]
        
        for ext in extensions:
            images.update(
                glob.glob(
                    os.path.join(self.original_data.dir_images, os.path.join("**", ext)),
                    recursive= True
                )
            )
        
        for image in images:
            name_image = os.path.splitext(os.path.basename(image))[0]
            if name_image in anomalies:
                images_to_remove.add(image)
        images -= images_to_remove
        
        for path in tqdm(images, desc= DESC_PROGRESSBAR_MASKS, disable= not verbose):
            with Image.open(path) as img:
                resizeImgAndSave(
                    img,
                    os.path.join(
                        self.new_data.dir_images,
                        os.path.splitext(os.path.basename(path))[0] + '.jpg'
                    ),
                    'JPEG',
                    new_size[0],
                    new_size[1]
                )
        return images

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

    def annotationsAnalysis(self):
        ae = AnnotationEncoder(self.original_data.dir_annotations)
        return ae.chooseLabels()

    def patch(self, size_img= (None, None), names_labels= None, verbose= True, debug_annotations= False):
        # Compute encoding of the labels
        ae = AnnotationEncoder(self.original_data.dir_annotations)
        if names_labels:
            encoding_labels = ae.encodeLabels(names_labels)
        else:
            encoding_labels, names_labels = ae.encodeLabelsInteractive()

        # Extract the coordinates of the labels
        shapes_label_file, anomalies = ae.extractCoordsLabels(debug_annotations)
        
        # Debug annotation extractions
        if debug_annotations:
            AnnotationEncoder.debugCodesLabels(ae.codes_labels)
            AnnotationEncoder.debugCoordinates(shapes_label_file)

        # Group data by selected label
        dir_category = '_' + '_'.join([label for label in names_labels]) + '_'
        self.new_data.wrapDirsSelf(dir_category)
        self._ensureNewDataDirs_()
        
        # Copy images
        images = self._copyImages_(new_size= size_img, anomalies= anomalies, verbose= verbose)

        # Build label masks
        mb = MaskBuilder(self.new_data.dir_labels,
                         ae.codes_labels,
                         shapes_label_file)
        masks = mb.buildAllMasks(new_size= size_img, verbose= verbose)

        assert len(images) == len(masks)
        
        # Write data summary
        df = pd.DataFrame(data= zip(images, masks))
        df.to_csv(os.path.join(self.new_data.dir_data, "data.csv"), header=False)

        # Write encoding of the labels in the class file
        encoding_labels.colors = encoding_labels.colors[:len(encoding_labels.labels)]
        #encoding_labels.one_hot_encoding = encoding_labels.one_hot_encoding[:len(encoding_labels.labels)]
        encoding_labels.to_json(os.path.join(self.new_data.dir_data, "classfile.json"))
