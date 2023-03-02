"""
=======================================
XML Parser and Miner
=======================================

This operation module offers functions designed to parse XML files and extract specific data inside.

"""

from xml.etree import ElementTree


def _pathToXMLRoot_(path: str):
    """
    Get the root of the XML ElementTree of the file whose path is entered as a parameter.
    """
    tree = ElementTree.parse(path)
    return tree.getroot()


def _parseTagsNamespace_(path: str, tag_maps= dict()) -> dict:
    """
    Return all the XML tags in the XML file whose path is entered as a parameter.
    """
    root = _pathToXMLRoot_(path)
    for e in root.iter():
        tag, namespace = _cleanTag_(e)
        if tag not in tag_maps:
            tag_maps[tag] = [namespace]
        elif namespace not in tag_maps[tag]:
            tag_maps[tag].append(namespace)
    return tag_maps


def _addAttributeValuesFromTag_(tag, attribute, tag_map, list, root):
    for namespace in tag_map:
        for tag_root in root.findall(namespace + tag):
            list.append(extractAttributeElem(tag_root, attribute))
    return list


def _parseNamespace_(raw_tag: str):
    return raw_tag[:raw_tag.find('}')+1]


def _parseTag_(raw_tag: str):
    return raw_tag[raw_tag.find('}')+1:]


def _cleanTag_(elem: ElementTree):
    """
    Return the tag of the ElementTree whitout its namespace.
    """
    raw_tag = elem.tag
    return _parseTag_(raw_tag), _parseNamespace_(raw_tag)


def collectAllTags(paths_annotations: list, warning= True) -> list:
    """
    Return all the tags present in the annotations.
    """
    tag_maps = dict()
    for path in paths_annotations:
        try:
            _parseTagsNamespace_(path, tag_maps)
        except ElementTree.ParseError as p:
            if warning:
                print(f"WARNING: The XML file {path} could not be parsed because it is malformed.")
    return tag_maps


def extractAttributeElem(elem: ElementTree, attribute):
    return elem.get(attribute)


def extractTagElem(path: str, tag: str, tag_map: str):
    try:
        root = _pathToXMLRoot_(path)
        for namespace in tag_map:
            elem_tag = root.find(namespace + tag)
            if elem_tag:
                break
    except ElementTree.ParseError as p:
        raise Exception(f"WARNING: The XML file {path} could not be parsed because it is malformed.")
    return elem_tag


def extractAttributesTag(tag, attributes, perimeter_tags, tag_maps, root):
    dictionary = {pt : list() for pt in perimeter_tags}
    for pt in perimeter_tags:
        if pt in tag_maps:
            for namespace in tag_maps[pt]:
                for pt_elem in root.findall('.//' + namespace + pt):
                    _addAttributeValuesFromTag_(
                        tag,
                        attributes,
                        tag_maps[tag],
                        dictionary[pt],
                        pt_elem)
    return dictionary
