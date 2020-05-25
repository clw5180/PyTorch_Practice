训练说明
(1)执行git clone --depth=1 https://github.com/ultralytics/yolov3 得到最新版本训练代码；
(2)执行cp /mfs/home/dataset/xxx_detection/xxx_detect/yolov3/cfg/xxx* /mfs/home/dataset/yolov3/cfg/ 将布匹疵点的配置文件xxx.name、xxx.data、xxx_yolov3.cfg以及xxx_yolov3-spp.cfg拷贝到最新版本的配置目录下；
(3)执行cp /mfs/home/dataset/xxx_detection/xxx_detect/anchors/Step* /mfs/home/dataset/yolov3/ 拷贝kmeans脚本到项目下，
   执行python Step1_voc2kmeansTxt.py '/mfs/home/dataset/train_val/rgb/rawsize/train/'完成10次聚类，选择acc较高的结果对应的anchor，下一步会用到
(4)修改xxx.data中classes=34,即实际的类别数
   查看xxx.name类别定义文件，即当前定义的34种缺陷名称
   修改xxx_yolov3.cfg中3个位置(因为共有3个yolo层，因此有3个位置需要修改)的classes、anchors(将上一步的anchor值复制过来)、filters=117 (117=(34+4+1)x3)
(5)执行cp /mfs/home/dataset/xxx_detection/xxx_detect/yolov3/xml2txt.py /mfs/home/dataset/yolov3/ 之后执行python xml2txt.py '/mfs/home/dataset/train_val/rgb/rawsize/train/' '/mfs/home/dataset/train_val/rgb/rawsize/val/' 将xml格式的训练数据转换成txt格式的数据，以便之后训练使用；执行后yolov3目录下多了train.txt和valid.txt两个文件
(6)将本地的darknet53.conv.74模型权重文件拷贝到yolov3/weights下：cp /home/user/huminglong/xxx_detection/xxx_detect/yolov3/weights/darknet53.conv.74 /mfs/home/dataset/yolov3/weights/  (或者从网上下载,时间比较长：cd yolov3/weights然后wget -c https://pjreddie.com/media/files/darknet53.conv.74)
(7)训练环境准备，目前测试过使用1.1版本pytorch, 3.7版本python可以运行；57上面有可用的conda环境，可以直接在57执行conda activate open-mmlab
(8)训练，执行python train.py --epochs=60 --batch-size=4 --accumulate=1 --cfg='cfg/xxx_yolov3.cfg' --data='cfg/xxx.data' --img-size=1024 --weights='weights/darknet53.conv.74' (可以使用tensorboard --logdir='xxx'观察训练结果，xxx代表tensorboard所需的文件路径，默认放置在yolov3/run文件夹下;远程操作需要启动浏览器如firefox进行观察)
(9)测试mAP，执行python test.py --cfg='cfg/xxx_yolov3.cfg' --data='cfg/xxx.data' --img-size=1024 --weights='weights/last.pt' --iou-thres=0.3 --conf-thres=0.005 --nms-thres=0.5
(10)结果可视化，python detect.py --source='/mfs/home/dataset/val/' --cfg='cfg/xxx_yolov3.cfg' --data='cfg/xxx.data' --img-size=1024 --weights='weights/last.pt' --conf-thres=0.03 --nms-thres=0.3
