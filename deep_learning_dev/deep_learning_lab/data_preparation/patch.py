"""The patch module contains classes and functions to patch a dataset and transform it into a dhSegment framework compatible one 
containing the masks of the specified labels.

Classes:
    DataStructure: Description of a dataset in the file system.
    DataPatcher: The main class of the library that allows the user to transform a dataset into a dhSegment framework 
                 compatible one containing the masks of the specified labels.
    AnnotationEncoder: Inspect a directory of annotations files to select, extract, and encode labels and their features.
    MaskBuilder: Creates masks for each labeled region in the image, given the annotations.

Notes:
    Its interactive functionnality allows the user to design diverse training datasets according to the elements of content 
    they want to extract.
    For the moment, it can only patch datasets under the PAGE format.

"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
import os, glob

from dh_segment_torch.data import ColorLabels, all_one_hot_and_colors, get_all_one_hots

import deep_learning_lab.data_preparation.xml_parser as xml
from deep_learning_lab import logging


RESULT_DIR = "results" # "" if deactivated. Also useful for temporary storage on Google Collab

_LOGGER = logging.getLogger(__name__)

_DESC_PROGRESSBAR_IMAGES = "Building label masks "
_DESC_PROGRESSBAR_MASKS =  "Copying images       "



def _resizeImgAndSave(img: Image, path: str, format: str, width: int = None, height: int = None) -> None:
    """Resize the given PIL.Image `img` and save it to the specified `path` with the
    specified image file `format`.

    Args:
        img (PIL.Image): The image to be resized and saved.
        path (str): The path where the resized image will be saved.
        format (str): The format of the output image file.
        width (int, optional): The desired width of the output image. If not provided,
                               the height is used to calculate the width while maintaining the aspect ratio.
        height (int, optional): The desired height of the output image. If not provided,
                                the width is used to calculate the height while maintaining the aspect ratio.

    """
    
    # Resize the image
    if width or height:
        _LOGGER.debug("Training requires all images to be the same size so resizing")
        if height and not width:
            width = int((height / float(img.size[1])) * float(img.size[0]))
        elif not height:
            height = int((width / float(img.size[0])) * float(img.size[1]))
        img = img.resize((width, height), Image.ANTIALIAS)

    # Save the image in the specified format
    img.save(path, format= format)



class DataStructure:
    """A class that represents a dataset in the file system. 
    
    Attributes:
        dir_data (str): The path to the directory that contains the dataset.
        dir_images (str): The path to the directory that contains the image files.
        dir_labels (str, optional): The path to the directory that contains the label files. Defaults to None.
        dir_annotations (str, optional): The path to the directory that contains the annotation files. Defaults to None.
    
    """
    
    def __init__(self, dir_data: str, dir_images: str, dir_labels: str = None, dir_annotations: str = None):
        """Initializes a DataStructure object with the given directories.
        
        Args:
            dir_data (str): The path to the directory that contains the dataset.
            dir_images (str): The path to the directory that contains the image files.
            dir_labels (str, optional): The path to the directory that contains the label files. Defaults to None.
            dir_annotations (str, optional): The path to the directory that contains the annotation files. Defaults to None.
        
        """
        self.wrapDirs(
            dir_data,
            dir_images,
            dir_labels,
            dir_annotations
        )
        
    
    @staticmethod
    def collectPaths(dir, extension):
        """Collects the absolute paths of all files with a specified extension in a directory.
        
        Args:
            dir (str): The path to the directory.
            extension (str): The file extension to search for.
        
        Returns:
            A list of absolute paths to the files with the specified extension in the directory.

        """
        paths = glob.glob(os.path.join(dir, extension))
        assert len(paths)
        return sorted(paths)

    
    def wrapDirs(self, dir_data: str, dir_images: str, dir_labels: str, dir_annotations: str, child_dir_data: str = "") -> None:
        """Sets the attributes for the DataStructure object based on the given directories.
        
        Args:
            dir_data (str): The path to the directory that contains the dataset.
            dir_images (str): The path to the directory that contains the image files.
            dir_labels (str, optional): The path to the directory that contains the label files. Defaults to None.
            dir_annotations (str, optional): The path to the directory that contains the annotation files. Defaults to None.
            child_dir_data (str, optional): The child directory to append to the dir_data path. Defaults to "".

        Notes:
            If RESULT_DIR is defined, it is the wrapper of wrapper_dir_data and it is removed from previous insertions.

        """
        self.dir_data = os.path.join(dir_data, child_dir_data) if child_dir_data else dir_data
        self.dir_images = os.path.join(dir_data, dir_images)
        self.dir_labels = os.path.join(dir_data, dir_labels) if dir_labels else None
        self.dir_annotations = os.path.join(dir_data, dir_annotations) if dir_annotations else None
    
    
    def wrapDirsSelf(self, wrapper_dir_data: str) -> None:
        """Sets the attributes that wrap all the directories into `wrapper_dir_data`.
        
        Args:
            wrapper_dir_data (str): the path of the directory that will be used as the new parent directory for all
            other directories.
        
        Notes:
            If `RESULT_DIR` is defined, it is used as the new parent directory instead of `wrapper_dir_data`.
            In this case, `self.dir_data` is renamed to the basename of its original path, and `self.dir_images`,
            `self.dir_labels`, and `self.dir_annotations` are updated accordingly.
        
        """
        if RESULT_DIR:
            # Use RESULT_DIR as the new parent directory
            wrapper_dir_data = os.path.join(
                RESULT_DIR,
                wrapper_dir_data
            )
            self.dir_data = os.path.basename(self.dir_data)
            self.dir_images = os.path.join(self.dir_data, os.path.basename(self.dir_images))
            self.dir_labels = os.path.join(self.dir_data, os.path.basename(self.dir_labels)) if self.dir_labels else None
            self.dir_annotations = os.path.join(self.dir_data, os.path.basename(self.dir_annotations)) if self.dir_annotations else None
        
        # Call the wrapDirs method to update the directories
        self.wrapDirs(
            dir_data= wrapper_dir_data,
            child_dir_data= self.dir_data,
            dir_images= self.dir_images,
            dir_labels= self.dir_labels,
            dir_annotations= self.dir_annotations
        )
    
    
    def __str__(self) -> str:
        """Returns a string representation of the DataStructure object.

        Returns:
            A string representing the name of the dataset.

        Notes:
            If RESULT_DIR is defined, it removes the first directory from the path before extracting the name.
        
        """
        dirs = os.path.normpath(self.dir_data).split(os.sep)
        if RESULT_DIR and len(dirs) > 1:
            return os.sep.join(dirs[1:])
        else:
            return dirs[-1]



class DataPatcher:
    """A class that transforms a dataset into a dhSegment framework compatible one containing the masks of the specified
    labels.

    Attributes:
        original_data (DataStructure): The original data to be patched.
        new_data (DataStructure): The new data to be created.

    """

    def __init__(self, original_data: DataStructure, new_data: DataStructure = None):
        """Initializes the directories of data to be patched.

        Args:
            original_data (DataStructure): The original data to be patched.
            new_data (DataStructure): The new data to be created.
        
        Raises:
            ValueError: If the original data and the new data do not have compatible structure.
        
        """
        if  original_data.dir_annotations   is None     \
        and new_data                        is not None \
        and new_data.dir_labels             is not None :
            value_error = ValueError(f"The original data and the new data do not have compatible structure.")
            _LOGGER.error(value_error)
            raise value_error
            
        self.original_data = original_data
        self.new_data = new_data
        self._ensureOriginalDataDirs()
                
    
    @property
    def _existing_labels(self) -> list:
        """Gets a list of existing labels from the new_data directory.

        Returns:
            A list of existing label filenames without the extension.

        """
        filenames_existing_labels = map(
            lambda path: os.path.splitext(os.path.basename(path))[0],
            glob.glob(
                os.path.join(self.new_data.dir_labels, os.path.join("*.png"))
            )
        )
        return list(filenames_existing_labels)

        
    def _ensureOriginalDataDirs(self) -> None:
        """Ensure that the directories in the original DataStructure of the dataset to patch are correct.
        If the directory does not exist, an exception is raised.

        Raises:
            Exception: If the directory in the original DataStructure does not exist.
        
        """
        for dir in vars(self.original_data).values():
            if dir and not os.path.isdir(dir):
                exception = Exception(f"{dir} does not exist.")
                _LOGGER.error(exception)
                raise exception

                
    def _ensureNewDataDirs(self) -> None:
        """Ensure that the directories of the new DataStructure of the patched dataset are created if they were not already
        created. If the directory does not exist, it is created.
        
        """
        for dir in vars(self.new_data).values():
            if dir:
                if not os.path.exists(dir):
                    _LOGGER.info(f"{dir} created")
                os.makedirs(dir, exist_ok= True)

                
    def _copyImages(self, new_size: tuple = (None, None), exceptions: set = set(), extensions= ['*.jpg', '*.tif', '*.png'],
                    verbose: bool = True) -> list:
        """Copy images from the original dataset to the new one, except those associated with exceptions.

        Args:
            new_size (tuple): A tuple (new_width, new_height) for the size of the image. Defaults to (None, None).
            exceptions (set): A set of strings representing the image file names to be excluded from copying. Defaults to
                              an empty set().
            extensions (iterable): An iterable of strings representing the file extensions of the images to be copied.
                                   Defaults to ['*.jpg', '*.tif', '*.png'].
            verbose (bool): A boolean flag to control the output. Defaults to True.

        Returns:
            A list of strings representing the image file paths that have been copied.
        
        """
        images = set()
        extensions += [ext.upper() for ext in extensions]
        
        for ext in extensions:
            images.update(
                glob.glob(
                    os.path.join(self.original_data.dir_images, os.path.join("**", ext)),
                    recursive= True
                )
            )
            
        images = list(filter(
            lambda image: not os.path.splitext(os.path.basename(image))[0] in exceptions,
            images
        ))
        
        for path in tqdm(images, desc= _DESC_PROGRESSBAR_MASKS, disable= not verbose):
            with Image.open(path) as img:
                _resizeImgAndSave(
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

    
    def patch(self, names_labels, size_img: tuple, overwrite: bool = True, verbose: bool = True, debug_annotations: bool = False):
        """Patch the original dataset by resizing and copying its images and by building the masks from its annotations.

        Args:
            names_labels (iterable): An iterable of label names for which to create masks.
            size_img (tuple): The size to which images should be resized.
            overwrite (bool, optional): Whether to overwrite existing images and masks. Defaults to True.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.
            debug_annotations (bool, optional): Whether to print debug messages about annotation extraction. Defaults to False.

        Raises:
            AssertionError: If the number of images and masks does not match.

        Returns:
            None.

        Notes:
            This method resizes and copies the original images and builds masks from the annotations for the specified labels.
            If the labels of the masks are not specified, the user is asked to choose them.
            Coordinates of the labels are then extracted from the annotations.
            The new dataset is created and filled with valid resized original images and their corresponding masks.
            A summary of the data and of the label encodings is finally created.

        Todo:
            Sort names_labels when using multilabels
            Use one-hot-encoding when using multilabels

        """

        # Compute encoding of the labels
        ae = AnnotationEncoder(self.original_data.dir_annotations)
        encoding_labels = ae.encodeLabels(names_labels)

        # Extract the coordinates of the labels
        shapes_label_name, anomalies = ae.extractCoordsLabels()
        
        # Debug annotation extractions
        if debug_annotations:
            AnnotationEncoder.debugCoordinates(shapes_label_name)
        _LOGGER.info(f"Color codes: {list(ae.codes_labels.items())}")
            
        # Group data by selected label
        #names_labels = names_labels.sort() # in the case of multilabels
        dir_category = '_'.join(names_labels)
        self.new_data.wrapDirsSelf(dir_category)
        self._ensureNewDataDirs()
        
        # Avoid overwrite images and labels
        if not overwrite:
            exceptions = self._existing_labels
            shapes_label_name = dict(filter(
                lambda item: item[0] not in exceptions, 
                shapes_label_name.items()
            ))
            anomalies.update(exceptions)
        
        # Stop the method if there is no data to patch
        if len(shapes_label_name) == 0:
            warning_msg = f"No new data to patch for `{', '.join(names_labels)}`"
            if verbose: print(warning_msg)
            _LOGGER.warning(warning_msg)
            return
        
        # Copy images
        images = self._copyImages(new_size= size_img, exceptions= anomalies, verbose= verbose)
        
        # Build label masks
        mb = MaskBuilder(self.new_data.dir_labels,
                         ae.codes_labels,
                         shapes_label_name)
        masks = mb.buildAllMasks(new_size= size_img, verbose= verbose)

        assert len(images) == len(masks)

        # Write encoding of the labels in the class file
        encoding_labels.colors = encoding_labels.colors[:len(encoding_labels.labels)]
        #encoding_labels.one_hot_encoding = encoding_labels.one_hot_encoding[:len(encoding_labels.labels)]
        encoding_labels.to_json(os.path.join(self.new_data.dir_data, "classfile.json"))
        
        _LOGGER.info(f"{self.original_data} patched for labels: {names_labels}")



class AnnotationEncoder:
    """A class for selecting, extracting and encoding labels and their features from a directory of annotation files.

    Attributes:
        dir_annotations (str): The directory path where the annotation files are stored.
        files (list): The list of annotation files collected from the directory.
        namespaces_label (dict): A dictionary of namespace prefixes and their corresponding URIs extracted from
                                 the annotation files.
        codes_labels (dict): A dictionary of label codes and their corresponding label names. If no labels are
                             encoded yet, it is None.

    """

    def __init__(self, dir_annotations: str):
        """Initializes an instance of AnnotationEncoder by collecting the paths of the annotation files,
        extracting the namespaces of the label tags from the files, and setting the codes_labels attribute to None.

        Args:
            dir_annotations (str): The directory path where the annotation files are stored.
        
        """
        self.files = DataStructure.collectPaths(dir_annotations, '*.xml')
        self.namespaces_label = xml.collectAllTags(self.files)
        self.codes_labels = None

        
    @staticmethod
    def _getPageSize(page):
        """Get the width and height of the annotation page.

        Args:
            page: The annotation page element obtained using xml_parser module.

        Returns:
            A tuple containing the width and height of the annotation page.
        
        Raises:
            Exception: If the page element does not have size value.
        
        """
        try:
            width = int(xml.extractAttributeElem(page, 'imageWidth'))
            height = int(xml.extractAttributeElem(page, 'imageHeight'))
        except Exception:
            raise Exception(f"WARNING: The Page attribute has no size value.")
        return width, height
    
    
    def _securelyExtractUsing(self, extraction_fun):
        """Securely extract data from XML files.

        Args:
            extraction_fun: The function to extract data from the XML.

        Returns:
            A tuple containing a dictionary with the extracted data and a set of anomalies.

        Warnings:
            Anomalies in annotation files will be logged using the '_LOGGER.warning' method.
        
        """
        result = dict()
        anomalies = set()
        for path in self.files:
            filename = os.path.splitext(os.path.basename(path))[0]
            try:
                page = xml.extractTagElem(path, 'Page', self.namespaces_label['Page'])
                result[filename] = extraction_fun(page)
            except Exception as e:
                anomalies.add(filename)
        if anomalies: _LOGGER.warning(f"Anomalies in annotation files: {', '.join(anomalies)}")
        return result, anomalies
    
    
    def calculateStatistics(self, names_labels):
        """Calculates statistics for the given names and labels.

        Args:
            names_labels (dict): A dictionary containing file names as keys and a list of labels for each file as values.

        Returns:
            dict: A dictionary containing statistics for each file and the following keys:
            - N_UNIQUE_LABELS (int): The number of unique labels across all files.
            - N_LABELS (int): The total number of labels across all files.
            - LABELS (dict): A dictionary containing the count of each label across all files.
        
        """
        stats_files = dict()
        unique_labels = set()
        number_labels_name = self.countLabelsFiles(names_labels)
        number_labels_all = dict.fromkeys(names_labels, 0)

        for file, number_labels in number_labels_name.items():
            number_labels = dict(sorted(number_labels.items(), key= lambda item: item[1], reverse= True))
            unique_labels.update(number_labels.keys())
            stats_files[file] = {
                "N_UNIQUE_LABELS": len(number_labels),
                "N_LABELS": sum(number_labels.values()),
                "LABELS": number_labels
            }
            for label, number in number_labels.items():
                number_labels_all[label] += number

        all_labels = sum(
            (list(number_labels.values()) for number_labels in number_labels_name.values()),
            []
        )
        stats_files = dict(sorted(stats_files.items(), key= lambda item: item[1]["N_LABELS"], reverse= True))
        stats_files["N_UNIQUE_LABELS"] = len(unique_labels)
        stats_files["N_LABELS"] = sum(all_labels)
        stats_files["LABELS"] = number_labels_all
        return stats_files
    
    
    def countLabelsFiles(self, names_labels) -> dict:
        """Counts the number of occurrences of specified label names in the given XML page.

        Args:
            names_labels (iterable): An iterable of label names to search for.

        Returns:
            The number of occurrences of the specified label names in the XML page.
        
        """
        number_labels_name, _ = self._securelyExtractUsing(
            lambda page: xml.countTags(
                page,
                names_labels,
                self.namespaces_label
            )
        )
        return number_labels_name

    
    def extractCoordsLabels(self) -> dict:
        """Extracts the coordinates of specified label names from the XML page.

        Returns:
            A dictionary containing the size of the page and the coordinates of the specified labels.
            A list of any anomalies encountered during extraction.
        
        """
        shapes_label_name, anomalies = self._securelyExtractUsing(
            lambda page: {
                'size': self._getPageSize(page),
                'coords': xml.extractAttributesTag(
                    'Coords',
                    'points',
                    page,
                    self.codes_labels.keys(),
                    self.namespaces_label
                )
            }
        )
        return shapes_label_name, anomalies
    
    
    def cleanAbsentLabels(self, labels):
        """Removes any labels from the given list that do not have a corresponding namespace in the class's dictionary
        of label namespaces.

        Args:
            labels (iterable): An iterable of label names.

        """
        labels_before = set(labels)
        for label in labels:
            if label not in self.namespaces_label.keys():
                labels.remove(label)
        if len(labels_before) > len(labels):
            _LOGGER.info(f"No label found for {labels_before - set(labels)}")
    
    
    def chooseLabels(self, preselection= []):
        """Asks the user to enter names of labels in a hierarchical order and returns them.
        
        Args:
            preselection (iterable): An iterable of preselected labels. Default is an empty list.

        Returns:
            A list of chosen labels in a hierarchical order.
        
        Warnings:
            Names of labels must be entered in a hierarchical order.
        
        """
        chosen_labels = []
        valid = False
        input_msg = "> "
        if preselection:
            input_msg += " ".join(preselection)
        
        print("Enter your labels in a hierarchical order separated by a space ' ':")
        for tag in self.namespaces_label: print(tag)
        print()

        # Input targeted labels
        while not valid:
            selection = input(input_msg)
            selection = list(filter(None, selection.strip(',\'').split(' ')))
            chosen_labels = preselection + selection
            
            valid = True
            for label in chosen_labels:
                valid &= (label in self.namespaces_label.keys())
                if not valid:
                    print("Parsing did not work. Please just copy-paste from the list above.")
                    break
        _LOGGER.info(f"User chose labels: {', '.join(chosen_labels)}")
        return chosen_labels

    
    def encodeLabels(self, names_labels, add_background: bool = True) -> dict:
        """Update the dictionary that encodes each label with a color and a one-hot encoding.

        Args:
            names_labels (iterable): An iterable of label names to encode.
            add_background (bool): Whether to add a 'Background' label with a default color and one-hot encoding. Default is True.

        Returns:
            A dictionary with each label as a key and a dictionary with 'color' and 'onehot' keys as the value.
        
        Todo:
            Edit to allow multilabels (single pixels that have multiple labels).

        """
        if add_background and 'Background' not in names_labels:
            names_labels = ['Background'] + names_labels
        
        #encoding_labels = _constantColorLabels.from_labels_multilabel(names_labels) # TODO Manage multilabels
        encoding_labels = _constantColorLabels.from_labels(names_labels)
        codes_labels = dict()
        
        for index, label in enumerate(encoding_labels.labels): # TODO Add multilabels
            codes_labels[label] = {'color': encoding_labels.colors[index]}
                                   #'onehot': encoding_labels.one_hot_encoding[index]}
        self.codes_labels = codes_labels
        return encoding_labels

    
    def encodeLabelsInteractive(self):
        """Update the dictionary that encodes each label chosen by the user with a color and a one-hot encoding.

        Returns:
            A tuple containing the updated label encoding dictionary and a list of the chosen label names.
        
        """
        names_labels = self.chooseLabels()
        encoding_labels = self.encodeLabels(names_labels)
        return encoding_labels, names_labels

    
    @staticmethod
    def debugCoordinates(shapes_label_name: dict) -> None:
        """Print the found coordinates for each label.

        Args:
            shapes_label_name (dict): A dictionary with each label as a key and a dictionary with 'coords' and 'size' keys
                                      as the value.
        
        """
        print("Found coordinates for:")
        for path, coords_label in shapes_label_name.items():
            print(path)
            for label, coords in coords_label['coords'].items():
                if label == 'Background':
                    width, height = coords_label['size']
                    print(f"({width}x{height})")
                else:
                    print('\t', label, coords)
                    print('\t', f"({len(coords)} object{'s' if len(coords)-1 else ''})")



class MaskBuilder:
    """A class for building image masks.

    Attributes:
        dir_masks (str): The directory path to save the generated masks.
        codes_labels (dict): A dictionary containing the mapping of shape labels to their corresponding colors.
        shapes_label_name (dict): A dictionary containing the shapes labels as keys and their corresponding coordinates
                                  and size as values.
        masks (list): A list of the paths of the generated masks.
    
    """

    def __init__(self, dir_masks: str, codes_labels: dict, shapes_label_name: dict):
        """Initializes the MaskBuilder object.

        Args:
            dir_masks (str): The directory path to save the generated masks.
            codes_labels (dict): A dictionary containing the mapping of shape labels to their corresponding colors.
            shapes_label_name (dict): A dictionary containing the shapes labels as keys and their corresponding coordinates
                                      and size as values.

        """
        self.dir_masks = dir_masks
        self.codes_labels = codes_labels
        self.shapes_label_name = shapes_label_name
        self.masks = []

        
    def _buildMask(self, coords_label: dict, original_size: tuple, new_size: tuple, name: str):
        """Builds a mask for the given shape coordinates with the specified original and new sizes.
        The generated mask is saved in the specified directory.

        Args:
            coords_label (dict): A dictionary containing the coordinates of the shape to be masked.
            original_size (tuple): A tuple of integers representing the original size of the image.
            new_size (tuple): A tuple of integers representing the new size of the image.
            name (str): The name to be used when saving the generated mask.

        """
        canvas = Image.fromarray(np.ones((original_size[1], original_size[0], 3), dtype='uint8'))
        draw = ImageDraw.Draw(canvas)
        for label, coords in coords_label.items():
            for object in coords:
                self.drawObject(object, self.codes_labels[label]['color'], draw)
        
        name += ".png"
        path_mask = os.path.join(self.dir_masks, name)
        self.masks.append(path_mask)
        _resizeImgAndSave(canvas, path_mask, 'PNG', new_size[0], new_size[1])

        
    @staticmethod
    def drawObject(points, colour, draw):
        """Draws the polygon object with the given points and color onto the ImageDraw object.

        Args:
            points (str): A string containing the coordinates of the object polygon.
            colour (str): The color to be used for drawing the object.
            draw (ImageDraw): The ImageDraw object onto which the object will be drawn.

        """
        points = [tuple(int(xy) for xy in point.split(',')) for point in points.split(' ')]
        draw.polygon(points, fill= colour)

        
def buildAllMasks(self, new_size: tuple, verbose: bool=True) -> list:
        """Builds a mask for all shapes in the shapes_label_name dictionary with the specified new size and returns the list
        of generated masks.

        Args:
            new_size (tuple): A tuple of integers representing the new size of the images.
            verbose (bool, optional): A flag indicating whether to display a progress bar during the generation of the masks.

        Returns:
            A list containing the paths of the generated masks.
        
        """
        for name, shapes_label in tqdm(self.shapes_label_name.items(), desc= _DESC_PROGRESSBAR_IMAGES, disable= not verbose):
            self._buildMask(shapes_label['coords'], shapes_label['size'], new_size, name)
        return self.masks



def _n_colors(n: int):
    """
    Returns a list of n random RGB colors generated with a constant seed.

    Args:
        n (int): The number of colors to generate.

    Returns:
        list: A list of n tuples representing RGB colors. Each tuple contains three integers between 0 and 255 inclusive,
              representing the red, green, and blue values of the color.

    Raises:
        None.
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


class _constantColorLabels(ColorLabels):
    """Overloaded class of ColorLabels from dhSegment that always returns the same colors for the labels.
    
    Args:
        ColorLabels (class): The base class for _constantColorLabels.

    Methods:
        from_labels(cls, labels): Returns an instance of ColorLabels of atomic labels.
        from_labels_multilabel(cls, labels): Returns an instance of ColorLabels of multi-labels.
    
    """

    @classmethod
    def from_labels(cls, labels):
        """Returns an instance of ColorLabels of atomic labels.
        
        Args:
            cls (class): The class object that the method is bound to.
            labels (iterable): An iterable of atomic labels.
        
        Returns:
            An instance of ColorLabels with the atomic labels.
        
        """
        num_classes = len(labels)
        colors = _n_colors(num_classes)
        for index, label in enumerate(labels):
            if label == 'Background':
                colors[index] = (0, 0, 0)
                break
        return cls(colors, labels=labels)
    
    
    @classmethod
    def from_labels_multilabel(cls, labels):
        """Returns an instance of ColorLabels of multi-labels.
        
        Args:
            cls (class): The class object that the method is bound to.
            labels (iterable): An iterable of atomic labels.
        
        Returns:
            An instance of ColorLabels with the multi-labels and their hot-encoding.
        
        """
        num_classes = len(labels)

        num_tries_left = 10
        while num_tries_left:
            num_tries_left -= 1
            base_colors = _n_colors(num_classes)
            for index, label in enumerate(labels):
                if label == 'Background':
                    colors[index] = (0, 0, 0)
                    break
            one_hot_encoding, colors = all_one_hot_and_colors(base_colors)
            if len(colors) == len(set(colors)):
                break
        else:
            _LOGGER.warning(
                f"Could not find a color combination for {num_classes}. "
                "Falling back on one color per one hot encoding."
            )
            one_hot_encoding = get_all_one_hots(num_classes).tolist()
            colors = _n_colors(len(one_hot_encoding))
        return cls(colors, one_hot_encoding, labels)
