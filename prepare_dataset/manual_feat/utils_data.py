# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import os
import random
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
# from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree

import numpy as np

from prepare_dataset.tools.decorators import (countcall, dataclass, logger,
                                              timeit)


@logger
def save_lable_file(xml_file, dst_label_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    y = root.find("evaluation").attrib["evaluation"]
    with open(dst_label_path, "w", encoding="utf8") as fy:
        fy.write(y)


@logger
def get_label_from_index(labels, ids, id):
    """find the label value from index"""
    print("id: ", id)
    ind = ids.index(id)
    label = labels[ind]
    return label


def get_label_info(folder_patient, folder_date, label):
    """prepare info for a label to save in a file"""
    params = {}
    folder_patient = os.path.basename(folder_patient)
    folder_date = os.path.basename(folder_date)
    params["beforeSurgery"] = "True"
    params["idPatient"] = str(folder_patient)
    params["valid"] = "True"
    params["Date"] = str(folder_date)
    params["evaluation"] = str(label)
    return params


@logger
def get_dtime(current_time, last_time):
    dtime = current_time - last_time + 0.0000001
    if dtime > 1:
        dtime = last_time
    print("dTimes: ", dtime)
    return dtime


def find_time(img_file_name):
    """convert extracted time to second."""
    data = img_file_name.split("/")[-1:][0].split("\\")[-1:][0]
    end = data.replace(".jpg", "")
    mins = end.split("_")
    timeSec = (
        int(mins[0]) * 3600 + int(mins[1]) * 60 + int(mins[2]) + int(mins[3]) / 1000.0
    )
    return timeSec


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def prepare_xml_for_labels(params, path_to_save="items.xml"):
    """create xml file for labels"""
    data = ET.Element("measurementInfo")
    data.set("beforeSurgery", params["beforeSurgery"])
    data.set("idPatient", params["idPatient"])
    data.set("valid", params["valid"])

    Date_ = ET.SubElement(data, "measurementDate")
    Date_.set("Date", params["Date"])

    evaluation_ = ET.SubElement(data, "evaluation")
    evaluation_.set("evaluation", params["evaluation"])

    # create a new XML file with the results
    myfile = open(path_to_save, "w", encoding="utf8")
    mydata = prettify(data)
    myfile.write(mydata)


def get_labels_ids_from_csv(path_csv_file):
    """find labels and ids from csv file"""
    with open(path_csv_file, "r", encoding="utf8") as file:
        list_of_ids = []
        list_of_labels = []
        lines = file.readlines()[1:]
        for line in lines:  # read rest of lines
            line = [int(x) for x in line.split(",")]
            list_of_ids.append(line[0])
            list_of_labels.append(line[1])

    return list_of_ids, list_of_labels


@logger
def merge_and_save_files(files, dst_data_path):
    # data = data2 = ""
    # Reading data from file1
    with open(dst_data_path, "w", encoding="utf8") as fw:
        for file in files:
            with open(file, "r", encoding="utf8") as fr:
                data = fr.read()
                fw.write(data)


# prepare the list of features to be saved in the text file
def prepare_list_to_save_in_file(p_arr):
    """make array of features ready to save into file"""
    p_list = [str(item) for item in p_arr]
    # p_list = [item for sublist in p_str for item in sublist]
    p_list_to_file = ["%s" % item for item in p_list]
    p_list_to_file = ",".join(p_list_to_file)
    return p_list_to_file


def write_lists_to_files(list1, list2, path1, path2):
    with open(path1, "w", encoding="utf8") as f:
        for item in list1:
            # write each item on a new line
            f.write("%s\n" % item)
        print("Done")

    with open(path2, "w", encoding="utf8") as f:
        for item in list2:
            # write each item on a new line
            f.write("%s\n" % item)
        print("Done")


def add_zero_padding(n_kp):
    """add zeros to empty rows from windows size"""
    # make zero array with the size of coordinate of keypoints (x,y) plus dtime
    padd = np.zeros(2 * n_kp + 1)
    padd = [str(val) for val in padd]
    padd = ",".join(padd)
    return padd
