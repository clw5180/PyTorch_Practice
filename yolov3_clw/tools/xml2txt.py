# -*- coding=utf-8 -*-
"""
该代码库使用yolo3的官方数据:
+labels
    +train2014
        -pic1.txt
        ...
    +val2014
        -pic2.txt
        ...
    -trainvalno5k.txt
    -5k.txt

每个label文件 pic1.txt:　多行,1行1个obj,如:56 0.855422 0.633625 0.087594 0.208542　　　小数,且为xywh格式


trainvalno5k.txt和5k.txt里每行是1个pic图片的绝对路径,如 /home/xy/test/test/test/images/val2014/COCO_val2014_000000581899.jpg
images与上面的labels同级.

故 coco如下:
+coco
    +annotations
    +images
        +train2014
        +val2014
    +labels
        +train2014
        +val2014
        -trainvalno5k.txt
        -5k.txt

该脚本将自己的数据集转换成如上形式. 
有data, 内含所有图片和xml.

+data
    -jpgs
    -xmls
->

+data
    +images
        -jpgs
        -xmls
    +labels
        -label1.txt
        ...
    -train.txt
    [-test.txt]
"""

import xml.etree.ElementTree as ET
import os
import sys
import shutil

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
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return filename, (width, height), coords


def write_single_label(xml_filepath, class2indice):
    "输入原xml的绝对路径"
    _, (width, height), coords = parse_xml(xml_filepath)

    #"56 0.855422 0.633625 0.087594 0.208542"    
    label = xml_filepath[:-3] + "txt"
    f = open(label, 'w')
    for i,coor in enumerate(coords):
        try:
            xmin, ymin, xmax, ymax, c = coor
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

    class2indice = {
        "car"    :0,
        "person" :1
    }

    if train_path is not None and os.path.exists(train_path):
        xml2txt(train_path, './train.txt',  class2indice)
    else:
        print("not exists '%s'" %(train_path))

    if valid_path is not None and os.path.exists(valid_path):
        xml2txt(valid_path, './valid.txt',  class2indice)
        xml2txt(test_path, './test.txt',  class2indice)
    else:
        print("not exists '%s' or '%s'" %(valid_path, test_path))

    print('done')
    pass

if __name__ == "__main__":

    train_path = '/home/caoliwei/train/'
    valid_path = '/home/caoliwei/val/'
    test_path = '/home/caoliwei/test/'

    if len(sys.argv) > 1:
        train_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        valid_path = sys.argv[2]

    if len(sys.argv) > 3:
        test_path = sys.argv[3]
    
    print("cloth_train_data(train_path='%s', valid_path='%s', test_path='%s')" % (train_path, valid_path, test_path))
    cloth_train_data(train_path, valid_path, test_path)
    pass
