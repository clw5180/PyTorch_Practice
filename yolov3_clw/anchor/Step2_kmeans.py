#coding:utf-8

import sys
import numpy as np
from collections import Counter


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
                xmax = int(infos[i].split(',')[2])
                xmin = int(infos[i].split(',')[0])
                width = xmax - xmin
                height = int(infos[i].split(',')[3]) - int(infos[i].split(',')[1])
                dataSet.append([width, height])

                # --------------------------------------------------------------------------
                # clw add: 统计所有box宽和高的最大最小值
                global box_width_min
                global box_height_min
                global  box_width_max
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
                box_scale_list.append(round(width/height, 2))
                #--------------------------------------------------------------------------

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
            
        print("\nK anchors: {}".format(format_anchors))
        print("Accuracy: {:.2f}%".format( self.avg_iou(all_boxes, result) * 100))
        pass

def kmeans_anchors(filename, cluster_number):
    ## 多次聚类
    for i in range(0, 10): #clw modify:多次聚类找到最佳结果
       kmeans = YOLO_Kmeans(cluster_number, filename)
       kmeans.txt2clusters()
    pass


if __name__ == "__main__":
    # 里面是一系列图片文件路径和标注信息,可以用voc2kemansTxt.py生成
    filename = "train.txt"
    cluster_number = 9  # anchor的个数,默认9

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    if len(sys.argv) > 2:        
        cluster_number = eval(sys.argv[2])

    kmeans_anchors(filename, cluster_number)
    pass
