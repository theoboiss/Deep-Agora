"""
=======================================
XML Parser and Extractor
=======================================

This operation module offers functions designed to parse XML files and extract specific data inside.

"""

from xml.etree import ElementTree
from collections import defaultdict


def _pathToXMLRoot(path: str):
    """
    Get the root of the XML ElementTree of the file whose path is entered as a parameter.
    """
    tree = ElementTree.parse(path)
    return tree.getroot()


def _parseTagsNamespace(path: str, namespaces_tag= defaultdict(set)) -> dict:
    """
    Return all the XML tags in the XML file whose path is entered as a parameter.
    """
    root = _pathToXMLRoot(path)
    for e in root.iter():
        tag, namespace = _cleanTag(e)
        namespaces_tag[tag].add(namespace)
    return namespaces_tag


def _addAttributeValuesFromTag(tag, attribute, root, namespaces, all_attributes):
    for namespace in namespaces:
        for tag_root in root.findall(namespace + tag):
            all_attributes.append(extractAttributeElem(tag_root, attribute))
    return all_attributes


def _cleanTag(elem: ElementTree):
    """
    Return the tag of the ElementTree whitout its namespace.
    """
    raw_tag = elem.tag
    end_namespace = raw_tag.find('}')+1
    return raw_tag[end_namespace:], raw_tag[:end_namespace]


def _iterateElementsPerimeter(perimeter_tags, namespaces_tag, root):
    """
    Create an iterator over the ElementTree of the perimeter tags that are covered by a namespace.
    """
    for pt in perimeter_tags:
        if pt in namespaces_tag:
            for namespace in namespaces_tag[pt]:
                for pt_elem in root.findall('.//' + namespace + pt):
                    yield pt_elem, namespace, pt
    

def countTags(root, perimeter_tags, namespaces_tag):
    counter_tags = defaultdict(int)
    for element, namespace, perimeter_tag in _iterateElementsPerimeter(perimeter_tags, namespaces_tag, root):
        tag, _ = _cleanTag(element)
        counter_tags[tag] += 1
    return counter_tags


def collectAllTags(paths_annotations: list, warning= True) -> list:
    """
    Return all the tags present in the annotations.
    """
    namespaces_tag = defaultdict(set)
    for path in paths_annotations:
        try:
            _parseTagsNamespace(path, namespaces_tag)
        except ElementTree.ParseError:
            if warning:
                print(f"WARNING: The XML file {path} could not be parsed because it is malformed.")
    return namespaces_tag


def extractAttributeElem(elem: ElementTree, attribute):
    return elem.get(attribute)


def extractTagElem(path: str, tag: str, namespaces: set):
    try:
        root = _pathToXMLRoot(path)
        for namespace in namespaces:
            elem_tag = root.find(namespace + tag)
            if elem_tag:
                break
    except ElementTree.ParseError:
        raise Exception(f"WARNING: The XML file {path} could not be parsed because it is malformed.")
    return elem_tag


def extractAttributesTag(tag, attributes, root, perimeter_tags, namespaces_tag):
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
