# DarkLabel转换MOT、ReID、VOC格式数据集脚本分享

上一篇文章推荐了DarkLabel标注软件，承诺会附上配套的代码，本文主要分享的是格式转换的几个脚本。

先附上脚本地址： https://github.com/pprp/SimpleCVReproduction/tree/master/DarkLabel 

先来了解一下为何DarkLabel能生成这么多格式的数据集，来看看DarkLabel的格式：

```python
frame(从0开始计), 数量, id(从0开始), box(x1,y1,x2,y2), class=null
0,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
1,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
2,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
```

每一帧，每张图片上的目标都可以提取到，并且每个目标有bbox、分配了一个ID、class

这些信息都可以满足目标检测、ReID、跟踪数据集。

ps：说明一下，以下脚本都是笔者自己写的，专用于单类的检测、跟踪、重识别的代码，如果有需要多类的，还需要自己修改多类部分的代码。 另外以下只针对Darklabel中**frame#,n,[,id,x1,y1,x2,y2,label]**格式。 

## 1. DarkLabel转Detection

这里笔者写了一个脚本转成VOC2007中的xml格式的标注，代码如下：

```python
import cv2
import os
import shutil
import tqdm
import sys

root_path = r"I:\Dataset\VideoAnnotation"

def print_flush(str):
    print(str, end='\r')
    sys.stdout.flush()

def genXML(xml_dir, outname, bboxes, width, height):
    xml_file = open((xml_dir + '/' + outname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + outname + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + 'cow' + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(x1) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(y1) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(x2) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(y2) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')


def gen_empty_xml(xml_dir, outname, width, height):
    xml_file = open((xml_dir + '/' + outname + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + outname + '.png' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    xml_file.write('</annotation>')


def getJPG(src_video_file, tmp_video_frame_save_dir):
    # gen jpg from video
    cap = cv2.VideoCapture(src_video_file)
    if not os.path.exists(tmp_video_frame_save_dir):
        os.makedirs(tmp_video_frame_save_dir)

    frame_cnt = 0
    isrun, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]

    while (isrun):
        save_name = append_name + "_" + str(frame_cnt) + ".jpg"
        cv2.imwrite(os.path.join(tmp_video_frame_save_dir, save_name), frame)
        frame_cnt += 1
        print_flush("Extracting frame :%d" % frame_cnt)
        isrun, frame = cap.read()
    return width, height

def delTmpFrame(tmp_video_frame_save_dir):
    if os.path.exists(tmp_video_frame_save_dir):
        shutil.rmtree(tmp_video_frame_save_dir)
    print('delete %s success!' % tmp_video_frame_save_dir)

def assign_jpgAndAnnot(src_annot_file, dst_annot_dir, dst_jpg_dir, tmp_video_frame_save_dir, width, height):
    # get coords from annotations files
    txt_file = open(src_annot_file, "r")

    content = txt_file.readlines()

    for line in content:
        item = line[:-1]
        items = item.split(',')
        frame_id, num_of_cow = items[0], items[1]
        print_flush("Assign jpg and annotion : %s" % frame_id)

        bboxes = []

        for i in range(int(num_of_cow)):
            obj_id = items[1 + i * 6 + 1]
            obj_x1, obj_y1 = int(items[1 + i * 6 + 2]), int(items[1 + i * 6 + 3])
            obj_x2, obj_y2 = int(items[1 + i * 6 + 4]), int(items[1 + i * 6 + 5])
            # preprocess the coords
            obj_x1 = max(1, obj_x1)
            obj_y1 = max(1, obj_y1)
            obj_x2 = min(width, obj_x2)
            obj_y2 = min(height, obj_y2)
            bboxes.append([obj_x1, obj_y1, obj_x2, obj_y2])

        genXML(dst_annot_dir, append_name + "_" + str(frame_id), bboxes, width,
            height)
        shutil.copy(
            os.path.join(tmp_video_frame_save_dir,
                        append_name + "_" + str(frame_id) + ".jpg"),
            os.path.join(dst_jpg_dir, append_name + "_" + str(frame_id) + ".jpg"))

    txt_file.close()


if __name__ == "__main__":
    append_names = ["cutout%d" % i for i in range(19, 66)]

    for append_name in append_names:
        print("processing",append_name)
        src_video_file = os.path.join(root_path, append_name + ".mp4")

        if not os.path.exists(src_video_file):
            continue

        src_annot_file = os.path.join(root_path, append_name + "_gt.txt")

        dst_annot_dir = os.path.join(root_path, "Annotations")
        dst_jpg_dir = os.path.join(root_path, "JPEGImages")

        tmp_video_frame_save_dir = os.path.join(root_path, append_name)

        width, height = getJPG(src_video_file, tmp_video_frame_save_dir)

        assign_jpgAndAnnot(src_annot_file, dst_annot_dir, dst_jpg_dir, tmp_video_frame_save_dir, width, height)

        delTmpFrame(tmp_video_frame_save_dir)
```

如果想转成U版yolo需要的格式可以点击 https://github.com/pprp/voc2007_for_yolo_torch 使用这里的脚本。

## 2. DarkLabel转ReID数据集

ReID数据集其实与分类数据集很相似，最出名的是Market1501数据集，对这个数据集不熟悉的可以先百度一下。简单来说ReID数据集只比分类中多了query, gallery的概念，也很简单。转换代码如下：

```python
import os
import shutil
import cv2
import numpy as np
import glob
import sys
import random
"""[summary]
根据视频和darklabel得到的标注文件
"""

def preprocessVideo(video_path):
    '''
    预处理，将视频变为一帧一帧的图片
    '''
    if not os.path.exists(video_frame_save_path):
        os.mkdir(video_frame_save_path)

    vidcap = cv2.VideoCapture(video_path)
    (cap, frame) = vidcap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    cnt_frame = 0

    while (cap):
        cv2.imwrite(
            os.path.join(video_frame_save_path, "frame_%d.jpg" % (cnt_frame)),
            frame)
        cnt_frame += 1
        print(cnt_frame, end="\r")
        sys.stdout.flush()
        (cap, frame) = vidcap.read()
    vidcap.release()
    return width, height


def postprocess(video_frame_save_path):
    '''
    后处理，删除无用的文件夹
    '''
    if os.path.exists(video_frame_save_path):
        shutil.rmtree(video_frame_save_path)


def extractVideoImgs(frame, video_frame_save_path, coords):
    '''
    抠图
    '''
    x1, y1, x2, y2 = coords
    # get image from save path
    img = cv2.imread(
        os.path.join(video_frame_save_path, "frame_%d.jpg" % (frame)))

    if img is None:
        return None
    # crop
    save_img = img[y1:y2, x1:x2]
    return save_img


def bbox_ious(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * \
                 (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area


def bbox_iou(box1, box2):
    # format box1: x1,y1,x2,y2
    # format box2: a1,b1,a2,b2
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    i_left_top_x = max(a1, x1)
    i_left_top_y = max(b1, y1)

    i_bottom_right_x = min(a2, x2)
    i_bottom_right_y = min(b2, y2)

    intersection = (i_bottom_right_x - i_left_top_x) * (i_bottom_right_y -
                                                        i_left_top_y)

    area_two_box = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1)

    return intersection * 1.0 / (area_two_box - intersection)


def restrictCoords(width, height, x, y):
    x = max(1, x)
    y = max(1, y)
    x = min(x, width)
    y = min(y, height)
    return x, y


if __name__ == "__main__":

    total_cow_num = 0

    root_dir = "./data/videoAndLabel"
    reid_dst_path = "./data/reid"
    done_dir = "./data/done"

    txt_list = glob.glob(os.path.join(root_dir, "*.txt"))
    video_list = glob.glob(os.path.join(root_dir, "*.mp4"))

    for i in range(len(txt_list)):
        txt_path = txt_list[i]
        video_path = video_list[i]

        print("processing:", video_path)

        if not os.path.exists(txt_path):
            continue

        video_name = os.path.basename(video_path).split('.')[0]
        video_frame_save_path = os.path.join(os.path.dirname(video_path),
                                             video_name)

        f_txt = open(txt_path, "r")

        width, height = preprocessVideo(video_path)

        print("done")

        # video_cow_id = video_name + str(total_cow_num)

        for line in f_txt.readlines():
            bboxes = line.split(',')
            ids = []
            frame_id = int(bboxes[0])

            box_list = []

            if frame_id % 30 != 0:
                continue

            num_object = int(bboxes[1])
            for num_obj in range(num_object):
                # obj = 0, 1, 2
                obj_id = bboxes[1 + (num_obj) * 6 + 1]
                obj_x1 = int(bboxes[1 + (num_obj) * 6 + 2])
                obj_y1 = int(bboxes[1 + (num_obj) * 6 + 3])
                obj_x2 = int(bboxes[1 + (num_obj) * 6 + 4])
                obj_y2 = int(bboxes[1 + (num_obj) * 6 + 5])

                box_list.append([obj_x1, obj_y1, obj_x2, obj_y2])
                # process coord
                obj_x1, obj_y1 = restrictCoords(width, height, obj_x1, obj_y1)
                obj_x2, obj_y2 = restrictCoords(width, height, obj_x2, obj_y2)

                specific_object_name = video_name + "_" + obj_id

                # mkdir for reid dataset
                id_dir = os.path.join(reid_dst_path, specific_object_name)

                if not os.path.exists(id_dir):
                    os.makedirs(id_dir)

                # save pic
                img = extractVideoImgs(frame_id, video_frame_save_path,
                                       (obj_x1, obj_y1, obj_x2, obj_y2))
                print(type(img))

                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    print(specific_object_name + " is empty")
                    continue

                # print(frame_id)
                img = cv2.resize(img, (256, 256))

                normalizedImg = np.zeros((256, 256))
                img = cv2.normalize(img, normalizedImg, 0, 255,
                                    cv2.NORM_MINMAX)

                cv2.imwrite(
                    os.path.join(id_dir, "%s_%d.jpg") %
                    (specific_object_name, frame_id), img)

            max_w = width - 256
            max_h = height - 256

            # 随机选取左上角坐标
            select_x = random.randint(1, max_w)
            select_y = random.randint(1, max_h)
            rand_box = [select_x, select_y, select_x + 256, select_y + 256]

            # 背景图保存位置
            bg_dir = os.path.join(reid_dst_path, "bg")
            if not os.path.exists(bg_dir):
                os.makedirs(bg_dir)

            iou_list = []

            for idx in range(len(box_list)):
                cow_box = box_list[idx]
                iou = bbox_iou(cow_box, rand_box)
                iou_list.append(iou)

            # print("iou list:" , iou_list)

            if np.array(iou_list).all() < 0:
                img = extractVideoImgs(frame_id, video_frame_save_path,
                                       rand_box)
                if img is None:
                    print(specific_object_name + "is empty")
                    continue
                normalizedImg = np.zeros((256, 256))
                img = cv2.normalize(img, normalizedImg, 0, 255,
                                    cv2.NORM_MINMAX)
                cv2.imwrite(
                    os.path.join(bg_dir, "bg_%s_%d.jpg") %
                    (video_name, frame_id), img)

        f_txt.close()
        postprocess(video_frame_save_path)
        shutil.move(video_path, done_dir)
        shutil.move(txt_path, done_dir)
```

数据集配套代码在： https://github.com/pprp/reid_for_deepsort  

## 3. DarkLabel转MOT16格式

其实DarkLabel标注得到信息和MOT16是几乎一致的，只不过需要转化一下，脚本如下：

```python
import os
'''
gt.txt:
---------
frame(从1开始计), id, box(left top w, h),ignore=1(不忽略), class=1(从1开始),覆盖=1), 
1,1,1363,569,103,241,1,1,0.86014
2,1,1362,568,103,241,1,1,0.86173
3,1,1362,568,103,241,1,1,0.86173
4,1,1362,568,103,241,1,1,0.86173

cutout24_gt.txt
---
frame(从0开始计), 数量, id(从0开始), box(x1,y1,x2,y2), class=null
0,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
1,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
2,4,0,450,194,558,276,null,1,408,147,469,206,null,2,374,199,435,307,null,3,153,213,218,314,null
'''


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # y = torch.zeros_like(x) if isinstance(x,
    #                                       torch.Tensor) else np.zeros_like(x)
    y = [0, 0, 0, 0]

    y[0] = (x[0] + x[2]) / 2
    y[1] = (x[1] + x[3]) / 2
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y

def process_darklabel(video_label_path, mot_label_path):
    f = open(video_label_path, "r")
    f_o = open(mot_label_path, "w")

    contents = f.readlines()

    for line in contents:
        line = line[:-1]
        num_list = [num for num in line.split(',')]

        frame_id = int(num_list[0]) + 1
        total_num = int(num_list[1])

        base = 2

        for i in range(total_num):

            print(base, base + i * 6, base + i * 6 + 4)

            _id = int(num_list[base + i * 6]) + 1
            _box_x1 = int(num_list[base + i * 6 + 1])
            _box_y1 = int(num_list[base + i * 6 + 2])
            _box_x2 = int(num_list[base + i * 6 + 3])
            _box_y2 = int(num_list[base + i * 6 + 4])

            y = xyxy2xywh([_box_x1, _box_y1, _box_x2, _box_y2])

            write_line = "%d,%d,%d,%d,%d,%d,1,1,1\n" % (frame_id, _id, y[0],
                                                        y[1], y[2], y[3])

            f_o.write(write_line)

    f.close()
    f_o.close()

if __name__ == "__main__":
    root_dir = "./data/videosample"

    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)

        video_path = os.path.join(full_path, item+".mp4")
        video_label_path = os.path.join(full_path, item + "_gt.txt")
        mot_label_path = os.path.join(full_path, "gt.txt")
        process_darklabel(video_label_path, mot_label_path)
```



---

DarkLabel软件可以到公众号GiantPandaCV后台回复“darklabel”获取。

以上就是DarkLabel转各种数据集格式的脚本了，DarkLabel还是非常方便的，可以快速构建自己的数据集。通常两分钟的视频可以生成2880张之多的图片，但是在目标检测中并不推荐将所有的图片都作为训练集，因为前后帧之间差距太小了，几乎是一模一样的。这种数据会导致训练速度很慢、泛化能力变差。

有两种解决方案：

- 可以选择隔几帧选取一帧作为数据集，比如每隔10帧作为数据集。具体选择多少作为间隔还是具体问题具体分析，如果视频中变化目标变化较快，可以适当缩短间隔；如果视频中大部分都是静止对象，可以适当增大间隔。
- 还有一种更好的方案是：对原视频用ffmpeg提取关键帧，将关键帧的内容作为数据集。关键帧和关键帧之间的差距比较大，适合作为目标检测数据集。

