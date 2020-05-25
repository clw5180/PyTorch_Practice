# coding:utf-8
# 功能：对 xml格式的数据集（训练集）进行聚类，找到合适的 anchor
# 输入：img_path
# 输出：save_path (no need to modify, default: 'train_info.txt')

import os
import xml.etree.ElementTree as ET
import sys
import numpy as np
from tqdm import tqdm
import argparse


# fliter_classes = ["car", "bus"]

def convert_annotation(xml_path):
    in_file = open(xml_path, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    annotations = ""
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls = cls.replace(' ', '_')  # 注意这一句非常关键,因为后面会按照空格提取txt内容,如果class name带空格,那么就会有bug

        ### 只对部分类别进行聚类
        # if cls not in fliter_classes:
        #     continue

        ### 如果前面已经决定 difficult obj 不参与训练，那么聚类的时候也应该过滤掉
        # difficult = obj.find('difficult').text
        # if int(difficult)==1:  # clw note:
        #    continue

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        annotations += " " + ",".join([str(a) for a in b]) + ',' + str(cls)
    return annotations

def scan_annotations(img_path, save_path = "train_info.txt"):
    image_names = [i for i in  os.listdir(img_path) if i.endswith(".png") or i.endswith(".jpg") ]
    list_file = open(save_path, 'w')
    pbar = tqdm(image_names)
    for image_name in pbar:
        pbar.set_description("Processing %s" % image_name)
        xml_path = os.path.join(img_path, image_name[:-4] + '.xml')  # 暂时认为 img 和 xml 在同一文件夹
        content = os.path.join(img_path, image_name) + convert_annotation(xml_path) + '\n'
        list_file.write(content)
    list_file.close()
    pass


box_width_min = 999999
box_height_min = 999999
box_width_max = -1
box_height_max = -1
box_width_list = []
box_height_list = []
box_scale_list = [] # clw note:宽高比

class YOLO_Kmeans:
    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            # 比如C:/Users/Administrator/Desktop/dataset_steer/JPEGImages/0009496A.jpg 407,671,526,771,0 378,757,502,855,0
            # 对应length=3
            length = len(infos)
            for i in range(1, length):  # clw note：这里要从1开始，因为0是图片路径字符串
                xmax = float(infos[i].split(',')[2])
                xmin = float(infos[i].split(',')[0])
                width = xmax - xmin
                height = float(infos[i].split(',')[3]) - float(infos[i].split(',')[1])
                dataSet.append([width, height])

                # --------------------------------------------------------------------------
                # clw add: 统计所有box宽和高的最大最小值
                global box_width_min
                global box_height_min
                global box_width_max
                global box_height_max

                if width < box_width_min:
                    box_width_min = width
                if height < box_height_min:
                    box_height_min = height
                if width > box_width_max:
                    box_width_max = width
                if height > box_height_max:
                    box_height_max = height

                box_width_list.append(width)
                box_height_list.append(height)
                box_scale_list.append(round(width / height, 2))
                # --------------------------------------------------------------------------

        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result_ratio = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)

        nAnchor = len(result_ratio)
        anchor = result_ratio[0]
        format_anchors = str(anchor[0]) + "," + str(anchor[1])
        for i in range(1, nAnchor):
            anchor = result_ratio[i]
            format_anchors += ",  " + str(anchor[0]) + "," + str(anchor[1])

        # print("\nK anchors: {}".format(format_anchors))
        # print("Accuracy: {:.2f}%".format( self.avg_iou(all_boxes, result) * 100))  # clw note
        # pass
        return format_anchors, self.avg_iou(all_boxes, result) * 100  # clw modify


def kmeans_anchors(filename, cluster_number):
    kmeans = YOLO_Kmeans(cluster_number, filename)
    anchors_max, acc_max = kmeans.txt2clusters()
    print('Multiple times kmeans and get best acc:')
    for i in tqdm(range(0, 10)):    # clw modify:多次聚类，比如聚类10次，输出最大的acc和对应的anchor
        kmeans = YOLO_Kmeans(cluster_number, filename)
        anchors, acc = kmeans.txt2clusters()
        if acc > acc_max:
            acc_max = acc
            anchors_max = anchors
    print("K anchors: {}".format(anchors_max))
    print("Accuracy: {:.2f}%".format(acc_max))  # clw note


if __name__ == "__main__":

    img_path = '/mfs/home/caoliwei/dataset/voc2007/val'  # 暂时认为 img 和 xml 在同一文件夹
    if not os.path.exists(img_path):
        print("not exists '%s'" %(img_path))
        sys.exit(0)

    save_path = 'train_info.txt'
    
    scan_annotations(img_path, save_path)  # 扫描所有xml，将训练集图片和对应box信息写入txt中
    kmeans_anchors(save_path, 9)           # 读取上面的txt，得到所有bbox，然后做聚类
    pass

