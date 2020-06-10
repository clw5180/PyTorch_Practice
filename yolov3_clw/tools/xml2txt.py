# -*- coding=utf-8 -*-
# 功能：(1)在xml_filepath对应路径下生成同名.txt文件，如 00001.txt 记录的坐标是归一化之后的
#       (2)记录所有图片的路径到 train.txt 或 valid.txt 文件中
# 输入：train_path， valid_path，test_path, class_name_path = '../cfg/coco.names'
#       train_save_path 默认为 '../train.txt'

import xml.etree.ElementTree as ET
import os
import sys
import shutil
from utils.parse_config import parse_data_cfg
from utils.utils import load_classes

train_path = '/home/user/dataset/voc2007/train'
#train_path = None
valid_path = '/home/user/dataset/voc2007/val'
test_path = None
class_name_path = '../cfg/voc.names' # 数据集类别名 list， 如 voc.names   coco.names


# 从xml文件中提取bounding box信息和w,h信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    tree = ET.parse(xml_path)		
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box.find('xmin').text)
        y_min = int(box.find('ymin').text)
        x_max = int(box.find('xmax').text)
        y_max = int(box.find('ymax').text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return filename, (width, height), coords


def write_single_label(xml_filepath, class2indice):
    "输入原xml的绝对路径"
    _, (width, height), coords = parse_xml(xml_filepath)

    #"56 0.855422 0.633625 0.087594 0.208542"    
    label = xml_filepath[:-4] + ".txt"
    f = open(label, 'w')
    for i,coor in enumerate(coords):
        try:
            xmin, ymin, xmax, ymax, c = coor
            # 有必要的话，应该做异常数据检测然后修正，如超出范围的box截取到边缘  TODO

            x = (xmin + xmax) / 2 / width
            y = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            ci = class2indice[c]
            line = [str(ci),str(x), str(y), str(w), str(h)]
            line = ' '.join(line) + "\n"
            f.write(line)
        except:
            print( 'write_single_label(', xml_filepath , '...) met unsupported class:', c)
            pass 
    f.close()
    pass


def xml2txt(image_path,output_txt_file_path, class2indice, append = False ,fix_JPG = True ):  
    '''
    '''  
    abs_image_path = os.path.abspath( image_path )

    # check if there is *.JPG 
    if fix_JPG:            
        for pos,_,fs in os.walk( abs_image_path ):
            for f in fs:
                if f.lower().endswith('jpg') and not f.endswith('jpg') :
                    src = os.path.join(pos,f)
                    des = os.path.join( pos, f[:-3] + "jpg" )
                    shutil.move(src,des)
                    
    lines = []
    for pos,_,fs in os.walk( abs_image_path ):
        for xml_file in fs:
            if xml_file.endswith('xml'):
                jpg_imgf = os.path.join(pos, xml_file[:-3] + "jpg")
                png_imgf = os.path.join(pos, xml_file[:-3] + "png")
                if os.path.exists( jpg_imgf ) : 
                    write_single_label( os.path.join(pos, xml_file) , class2indice )
                    lines.append( jpg_imgf )
                elif  os.path.exists( png_imgf ): 
                    write_single_label( os.path.join(pos, xml_file) , class2indice )
                    lines.append( png_imgf )

    if append:
        open(output_txt_file_path, 'a').write( '\n'.join(lines) )
    else:
        open(output_txt_file_path, 'w').write( '\n'.join(lines) )

    print('images number: ', len(lines))
    pass

def cloth_train_data(train_path=None, valid_path = None, test_path=None):
    ###
    class2indice={}
    names = load_classes(class_name_path)
    for i, name in enumerate(names):
        class2indice[name] = i
    ###

    if train_path is not None and os.path.exists(train_path):
        xml2txt(train_path, train_save_path,  class2indice)
    else:
        print("train path %s not exists !" %(train_path))

    if valid_path is not None and os.path.exists(valid_path):
        xml2txt(valid_path, valid_save_path,  class2indice)
    else:
        print("valid path %s not exists !" % (valid_path))

    if test_path is not None and os.path.exists(valid_path):
        xml2txt(test_path, test_save_path, class2indice)
    else:
        print("test path %s not exists !" % (test_path))

    print('done')


if __name__ == "__main__":

    # train_path = 'D:/dataset/train'
    # valid_path = 'D:/dataset/val'

    train_save_path = '../train.txt'
    valid_save_path = '../valid.txt'
    test_save_path = '../test.txt'

    if len(sys.argv) > 1:
        train_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        valid_path = sys.argv[2]

    if len(sys.argv) > 3:
        test_path = sys.argv[3]
    
    print("cloth_train_data(train_path='%s', valid_path='%s', test_path='%s')" % (train_path, valid_path, test_path))
    cloth_train_data(train_path, valid_path, test_path)
    pass
