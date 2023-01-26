"""
Operation module used to parse XML files.
"""

from xml.etree import ElementTree


def _pathToXMLRoot_(paths: list):
    """
    Get the root of the XML ElementTree of the file whose path is entered as a parameter.
    """
    tree = ElementTree.parse(paths)
    return tree.getroot()

def _collectTags_(path: str, tags= dict()) -> dict:
    """
    Return all the XML tags in the XML file whose path is entered as a parameter.
    """
    root = _pathToXMLRoot_(path)
    for e in root.iter():
        tag = cleanTag(e)
        if tag not in tags:
            tags[tag] = e.tag
    return tags

def _addAttributeValuesFromTag_(tag, attribute, list, root):
    for tag_root in root.findall(tag):
        list.append(extractAttributeElem(tag_root, attribute))
    return list

def cleanTag(elem: ElementTree):
    """
    Return the tag of the ElementTree whitout its namespace.
    """
    raw_tag = elem.tag
    return raw_tag[raw_tag.find('}')+1:]

def extractAttributeElem(elem: ElementTree, attribute):
    return elem.get(attribute)

def extractTagElem(path: str, tag: str):
    root = _pathToXMLRoot_(path)
    elem_tag = root.find(tag)
    return elem_tag

def collectAllTags(paths_annotations: list) -> list:
    """
    Return all the tags present in the annotations.
    """
    all_tags = dict()
    for pa in paths_annotations:
        _collectTags_(pa, all_tags)
    return all_tags

def extractAttributesTag(tag, attributes, perimeter_tags, tags_to_namespaces, root):
    dictionary = {tag : list() for tag in perimeter_tags}
    for pt in perimeter_tags:
        if pt in tags_to_namespaces:
            for pt_elem in root.findall('.//'+tags_to_namespaces[pt]):
                _addAttributeValuesFromTag_(
                    tags_to_namespaces[tag],
                    attributes,
                    dictionary[pt],
                    pt_elem)
    return dictionary
