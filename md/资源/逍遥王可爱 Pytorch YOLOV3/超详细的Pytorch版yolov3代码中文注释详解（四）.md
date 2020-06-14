本篇介绍如何让检测器在视频或者网络摄像头上实时工作。我们将引入一些命令行标签，以便能使用该网络的各种超参数进行一些实验。这个代码是video.py，代码整体上很像detect.py，只有几处变化，只是我们不会在 batch 上迭代，而是在视频的帧上迭代。

注意代码中有一处错误我进行了修改。源代码在计算scaling_factor时，用的scaling_factor =  torch.min(416/im_dim,1)[0].view(-1,1)显然不对，应该使用用户输入的args.reso即改为scaling_factor  = torch.min(int(args.reso)/im_dim,1)[0].view(-1,1)

接下来就开始吧。

```
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    视频检测模块的参数转换
    
    """
    #创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型
    parser = argparse.ArgumentParser(description='YOLO v3 检测模型')
    parser.add_argument("--bs", dest = "bs", help = "Batch size，默认为 1", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "目标检测结果置信度阈值", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS非极大值抑制阈值", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "配置文件",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "模型权重",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "网络输入分辨率. 分辨率越高,则准确率越高; 反之亦然",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "待检测视频目录", default = "video.avi", type = str)
    
    return parser.parse_args()
    
args = arg_parse()# args是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值   
# Namespace(bs=1, cfgfile='cfg/yolov3.cfg', confidence=0.5,det='det', images='imgs', nms_thresh=0.4, reso='416', weightsfile='yolov3.weights')
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()# GPU环境是否可用



num_classes = 80# coco 数据集有80类
classes = load_classes("data/coco.names")#将类别文件载入到我们的程序中，coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字



#初始化网络并载入权重
print("载入神经网络....")
model = Darknet(args.cfgfile)# Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
model.load_weights(args.weightsfile)# 将权重文件载入，并复制给对应的网络结构model中
print("模型加载成功.")
# 网络输入数据大小
model.net_info["height"] = args.reso# model类中net_info是一个字典。’’height’’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
inp_dim = int(model.net_info["height"])#inp_dim是网络输入图片尺寸（如416*416）
assert inp_dim % 32 == 0 # 如果设定的输入图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim > 32

# 如果GPU可用, 模型切换到cuda中运行
if CUDA:
    model.cuda()


#变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的
model.eval()


#要在视频或网络摄像头上运行这个检测器，代码基本可以保持不变，只是我们不会在 batch 上迭代，而是在视频的帧上迭代。
# 将方框和文字写在图片上
def write(x, results):
    c1 = tuple(x[1:3].int())# c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int())# c2为方框右下角坐标x2,y2
    img = results
    cls = int(x[-1])
    color = random.choice(colors)#随机选择一个颜色，用于后面画方框的颜色
    label = "{0}".format(classes[cls])#label为这个框所含目标类别名字的字符串
    cv2.rectangle(img, c1, c2,color, 1)# 在图片上画出(x1,y1,x2,y2)矩形，即我们检测到的目标方框
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]# 得到一个包含目标名字字符的方框的宽高
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4# 得到包含目标名字的方框右下角坐标c2，这里在x,y方向上分别加了3、4个像素
    cv2.rectangle(img, c1, c2,color, -1)# 在图片上画一个实心方框，我们将在方框内放置目标类别名字
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);# 在图片上写文字，(c1[0], c1[1] + t_size[1] + 4)为字符串的左下角坐标
    return img


#Detection phase

videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture(videofile) #用 OpenCV 打开视频

#cap = cv2.VideoCapture(0)  #for webcam(相机)

# 当没有打开视频时抛出错误
assert cap.isOpened(), 'Cannot capture source'
# frames用于统计图片的帧数
frames = 0  
start = time.time()

fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
fps = 24 
savedPath = './det/savevideo.avi' # 保存的地址和视频名
ret, frame = cap.read() 
videoWriter = cv2.VideoWriter(savedPath, fourcc, fps,(frame.shape[1], frame.shape[0])) # 最后为视频图片的形状

while cap.isOpened():# ret指示是否读入了一张图片，为true时读入了一帧图片
    ret, frame = cap.read()
    
    if ret:
        # 将图片按照比例缩放缩放，将空白部分用(128,128,128)填充，得到为416x416的图片。并且将HxWxC转换为CxHxW   
        img = prep_image(frame, inp_dim)
        #cv2.imshow("a", frame)
        # 得到图片的W,H,是一个二元素tuple.因为我们不必再处理 batch，而是一次只处理一张图像，所以很多地方的代码都进行了简化。
        #因为一次只处理一帧，故使用一个元组im_dim替代 im_dim_list 的张量。
        im_dim = frame.shape[1], frame.shape[0]
#先将im_dim变成长度为2的一维行tensor，再在1维度(列这个维度)上复制一次，变成1x4的二维行tensor[W,H,W,H]，展开成1x4主要是在后面计算x1,y1,x2,y2各自对应的缩放系数时好对应上。  
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)#repeat()可能会改变tensor的维度。它对tensor中对应repeat参数对应的维度上进行重复给定的次数，如果tensor的维度小于repeat()参数给定的维度，tensor的维度将变成和repeat()一致。这里repeat(1,2)，表示在第一维度上重复一次，第二维上重复两次，repeat(1,2)有2个元素，表示它给定的维度有2个,所以将长度为2的一维行tensor变成了维度为1x4的二维tensor   
                
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        # 只进行前向计算，不计算梯度
        with torch.no_grad():
#得到每个预测方框在输入网络图片(416x416)坐标系中的坐标和宽高以及目标得分以及各个类别得分(x,y,w,h,s,s_cls1,s_cls2...)
#并且将tensor的维度转换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)
            output = model(Variable(img, volatile = True), CUDA)
        #将方框属性转换成(ind,x1,y1,x2,y2,s,s_cls,index_cls)，去掉低分，NMS等操作，得到在输入网络坐标系中的最终预测结果
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)

        # output的正常输出类型为float32,如果没有检测到目标时output元素为0，此时为int型，将会用continue进行下一次检测
        if type(output) == int:
        #每次迭代，我们都会跟踪名为frames的变量中帧的数量。然后我们用这个数字除以自第一帧以来过去的时间，得到视频的帧率。
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
        #我们不再使用cv2.imwrite将检测结果图像写入磁盘，而是使用cv2.imshow展示画有边界框的帧。
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
        #如果用户按Q按钮，就会让代码中断循环，并且视频终止。
            if key & 0xFF == ord('q'):
                break
            continue
        
#im_dim一行对应一个方框所在图片尺寸。在detect.py中一次测试多张图片，所以对应的im_dim_list是找到每个方框对应的图片的尺寸。
# 而这里每次只有一张图片，每个方框所在图片的尺寸一样，只需将图片的尺寸的行数重复方框的数量次数即可                
        im_dim = im_dim.repeat(output.size(0), 1)
        # 得到每个方框所在图片缩放系数
        #scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)#这是源代码，下面是我修改的代码
        scaling_factor = torch.min(int(args.reso)/im_dim,1)[0].view(-1,1)
        # 将方框的坐标(x1,y1,x2,y2)转换为相对于填充后的图片中包含原始图片区域（如416*312区域）的计算方式。
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        # 将坐标映射回原始图片
        output[:,1:5] /= scaling_factor
        #将超过了原始图片范围的方框坐标限定在图片范围之内
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
        
        #coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字
        classes = load_classes('data/coco.names')
        #读入包含100个颜色的文件pallete，里面是100个三元组序列
        colors = pkl.load(open("pallete", "rb"))
        #将每个方框的属性写在图片上
        list(map(lambda x: write(x, frame), output))
        
        cv2.imshow("frame", frame)
        
        videoWriter.write(frame)           # 每次循环，写入该帧
        key = cv2.waitKey(1)
        # 如果有按键输入则返回按键值编码，输入q返回113
        if key & 0xFF == ord('q'):
            break
        #统计已经处理过的帧数
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        videoWriter.release()              # 结束循环的时候释放
        break     
```

