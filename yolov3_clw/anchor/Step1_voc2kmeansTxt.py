#coding:utf-8

import sys
import os
import xml.etree.ElementTree as ET
from os import getcwd
from Step2_kmeans import kmeans_anchors

classes = ["car", "person"]

#classes = ["dead cotton"]

def convert_annotation(xml_path):
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    annotations = ""
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        annotations += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
    return annotations

def scan_annotations(img_path, save_path = "train.txt"):
    image_names = [i for i in  os.listdir(img_path) if i.endswith(".png") or i.endswith(".jpg") ]
    list_file = open(save_path, 'w')
    for image_name in image_names:
        xml_path = os.path.join(img_path, image_name[:-4] + '.xml') 
        content = os.path.join(img_path, image_name) + convert_annotation(xml_path) + '\n'
        list_file.write(content)
    list_file.close()
    pass

if __name__ == "__main__":
    #img_path = "/mfs/home/textiledd/train_val/rgb/resize_4k_1k/resize_v5/train/"
    #img_path = '/mfs/home/caoliwei/textile/train/dead_cotton/' # clw note: path must not have space!!! because the generated train.txt will split(' ')
    #img_path = '/mfs/home/yinkunyang/train/'
    img_path = '/mfs/home/caoliwei/textile/train/3class/' # clw note: path must not have space!!! because the generated train.txt will split(' ')
    save_path = "train.txt"

    if len(sys.argv) > 1:
        img_path = sys.argv[1]

    if len(sys.argv) > 2:
        save_path = sys.argv[2]

    if not os.path.exists(img_path):
        print("not exists '%s'" %(img_path))
        sys.exit(0)

    scan_annotations(img_path, save_path)
    kmeans_anchors(save_path, 9)
    pass

