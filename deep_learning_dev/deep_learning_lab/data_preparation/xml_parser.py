"""Module providing functions for parsing XML files and extracting information from them.

Functions:
countTags(root, perimeter_tags, namespaces_tag): returns a dictionary with the count of each tag present in the
                                                 XML file that matches the perimeter tags and namespaces specified.
collectAllTags(paths_annotations): returns a dictionary with all the tags present in the annotations.
extractAttributeElem(elem, attribute): returns the value of an attribute in an ElementTree.
extractTagElem(path, tag, namespaces): returns an ElementTree that matches the tag and namespaces specified.
extractAttributesTag(tag, attributes, root, perimeter_tags, namespaces_tag): returns a dictionary with the values
                                                                             of the attributes for the specified tag
                                                                             and perimeter tags within the ElementTree.

Variables:
    _LOGGER: a logging instance for the module.

"""

from xml.etree import ElementTree
from collections import defaultdict

from deep_learning_lab import logging


_LOGGER = logging.getLogger(__name__)



def _pathToXMLRoot(path: str) -> ElementTree.Element:
    """Parses an XML file located at the given path and returns the root Element of the resulting ElementTree.

    Args:
        path (str): The path to the XML file to parse.

    Returns:
        ElementTree.Element: The root element of the parsed ElementTree.

    """
    tree = ElementTree.parse(path)
    return tree.getroot()


def _parseTagsNamespace(path: str, namespaces_tag= defaultdict(set)) -> dict:
    """Parses an XML file located at the given path and returns a dictionary containing all the tags present in the file.

    Args:
        path (str): The path to the XML file to parse.
        namespaces_tag (dict): A dictionary containing namespace information for the tags in the file.

    Returns:
        dict: A dictionary containing all the tags present in the file.

    """
    root = _pathToXMLRoot(path)
    for e in root.iter():
        tag, namespace = _cleanTag(e)
        namespaces_tag[tag].add(namespace)
    return namespaces_tag


def _addAttributeValuesFromTag(tag: str, attribute: str, root: ElementTree.Element, namespaces: set, all_attributes: list) -> list:
    """Adds the value of the specified attribute for all elements matching the given tag and namespace to a list.

    Args:
        tag (str): The tag to match.
        attribute (str): The attribute whose value to extract.
        root (ElementTree.Element): The root element of the ElementTree to search.
        namespaces (set): The namespaces to search for the tag in.
        all_attributes (list): The list to add the extracted attribute values to.

    Returns:
        list: The list of all extracted attribute values.

    """
    for namespace in namespaces:
        for tag_root in root.findall(namespace + tag):
            all_attributes.append(extractAttributeElem(tag_root, attribute))
    return all_attributes


def _cleanTag(elem: ElementTree.Element) -> tuple:
    """Returns the tag of an ElementTree.Element without its namespace.

    Args:
        elem (ElementTree.Element): The element whose tag to clean.

    Returns:
        tuple: A tuple containing the cleaned tag and the namespace it belonged to.

    """
    raw_tag = elem.tag
    end_namespace = raw_tag.find('}')+1
    return raw_tag[end_namespace:], raw_tag[:end_namespace]


def _iterateElementsPerimeter(perimeter_tags, namespaces_tag: dict, root: ElementTree.Element):
    """Creates an iterator over the ElementTree of the perimeter tags that are covered by a namespace.

    Args:
        perimeter_tags (iterable): An iterable of tags to search for.
        namespaces_tag (dict): A dictionary containing namespace information for the tags in the file.
        root (ElementTree.Element): The root element of the ElementTree to search.

    Yields:
        tuple: A tuple containing the matching element, the namespace it belongs to, and the perimeter tag it matches.

    """
    for pt in perimeter_tags:
        if pt in namespaces_tag:
            for namespace in namespaces_tag[pt]:
                for pt_elem in root.findall('.//' + namespace + pt):
                    yield pt_elem, namespace, pt
    

def countTags(root, perimeter_tags, namespaces_tag) -> dict:
    """Count the number of occurrences of each tag in the XML file whose root is entered as a parameter.

    Args:
        root (ElementTree): the root of the XML file.
        perimeter_tags (set): a set of tags representing the perimeter where the tags should be counted.
        namespaces_tag (dict): a dictionary containing the namespaces of each tag to be counted.

    Returns:
        A dictionary containing the count of each tag present in the XML file that matches the perimeter tags and namespaces specified.
    
    """
    counter_tags = defaultdict(int)
    for element, _, __ in _iterateElementsPerimeter(perimeter_tags, namespaces_tag, root):
        tag, _ = _cleanTag(element)
        counter_tags[tag] += 1
    return counter_tags


def collectAllTags(paths_annotations) -> list:
    """Return all the tags present in the annotations.

    Args:
        paths_annotations (iterable): An iterable of paths to the XML files containing the annotations.

    Returns:
        A dictionary containing all the tags present in the annotations.

    """
    namespaces_tag = defaultdict(set)
    for path in paths_annotations:
        try:
            _parseTagsNamespace(path, namespaces_tag)
        except ElementTree.ParseError:
            _LOGGER.warning(f"The XML file {path} could not be parsed because it is malformed.")
    return namespaces_tag


def extractAttributeElem(elem: ElementTree, attribute: str) -> str:
    """Extract the value of an attribute in an ElementTree.

    Args:
        elem (ElementTree): the ElementTree containing the attribute to be extracted.
        attribute (str): the name of the attribute to be extracted.

    Returns:
        The value of the attribute in the ElementTree.

    """
    return elem.get(attribute)


def extractTagElem(path: str, tag: str, namespaces: set) -> ElementTree:
    """Extract an ElementTree that matches the tag and namespaces specified.

    Args:
        path (str): the path to the XML file containing the ElementTree to be extracted.
        tag (str): the tag to be extracted.
        namespaces (set): the namespaces of the tag to be extracted.

    Returns:
        The ElementTree that matches the tag and namespaces specified.
    
    """
    try:
        root = _pathToXMLRoot(path)
        for namespace in namespaces:
            elem_tag = root.find(namespace + tag)
            if elem_tag:
                break
    except ElementTree.ParseError:
        raise Exception(f"The XML file {path} could not be parsed because it is malformed.")
    return elem_tag


def extractAttributesTag(tag: str, attributes, root: ElementTree, perimeter_tags: set, namespaces_tag: dict) -> dict:
    """Extract the values of the attributes for the specified tag and perimeter tags within the ElementTree.

    Args:
        tag (str): the tag containing the attributes to be extracted.
        attributes (iterable): the iterable of attributes to be extracted.
        root (ElementTree): the root of the ElementTree where the attributes will be extracted from.
        perimeter_tags (set): a set of tags representing the perimeter where the tags should be counted.
        namespaces_tag (dict): a dictionary containing the namespaces of each tag to be counted.

    Returns:
        A dictionary containing the values of the attributes for the specified tag and perimeter tags within the ElementTree.
    
    """
    attributes_perimeter = {pt : list() for pt in perimeter_tags}
    for element, namespace, perimeter_tag in _iterateElementsPerimeter(perimeter_tags, namespaces_tag, root):
        _addAttributeValuesFromTag(
            tag,
            attributes,
            element,
            namespaces_tag[tag],
            attributes_perimeter[perimeter_tag]
        )
    return attributes_perimeter
