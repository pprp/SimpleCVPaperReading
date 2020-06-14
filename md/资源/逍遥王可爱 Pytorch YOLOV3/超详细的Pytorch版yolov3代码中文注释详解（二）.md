本篇接着上一篇去解释util.py。这个程序包含了predict_transform函数（Darknet类中的forward函数要用到），write_results函数使我们的输出满足 objectness 分数阈值和非极大值抑制（NMS），以得到「真实」检测结果。还有prep_image和letterbox_image等图片预处理函数等（前者用来将numpy数组转换成PyTorch需要的的输入格式，后者用来将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充）。话不多说，直接看util.py的注释。

```
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):#因为同一类别可能会有多个真实检测结果，所以我们使用unique函数来去除重复的元素，即一类只留下一个元素，达到获取任意给定图像中存在的类别的目的。
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)#np.unique该函数是去除数组中的重复数字，并进行排序之后输出
    unique_tensor = torch.from_numpy(unique_np)
    # 复制数据
    tensor_res = tensor.new(unique_tensor.shape)# new(args, *kwargs) 构建[相同数据类型]的新Tensor
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    在特征图上进行多尺度预测, 在GRID每个位置都有三个不同尺度的锚点.predict_transform()利用一个scale得到的feature map预测得到的每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...),其中x,y,w,h
    是在网络输入图片坐标系下的值,s是方框含有目标的置信度得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率。输入的feature map(prediction变量) 
    维度为(batch_size, num_anchors*bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW存储方式。参数见predict_transform()里面的变量。
    并且将结果的维度变换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)的tensor，同时得到每个方框在网络输入图片(416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
    """
    batch_size = prediction.size(0)
    # stride表示的是整个网络的步长，等于图像原始尺寸与yolo层输入的feature mapr尺寸相除，因为输入图像是正方形，所以用高相除即可
    stride =  inp_dim // prediction.size(2)#416//13=32
    # feature map每条边格子的数量，416//32=13
    grid_size = inp_dim // stride
    # 一个方框属性个数，等于5+类别数量
    bbox_attrs = 5 + num_classes
    # anchors数量
    num_anchors = len(anchors)   
    # 输入的prediction维度为(batch_size, num_anchors * bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW
    # 存储方式，将它的维度变换成(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    #contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
    prediction = prediction.transpose(1,2).contiguous()
    # 将prediction维度转换成(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)。不看batch_size，
    # (grid_size*grid_size*num_anchors, bbox_attrs)相当于将所有anchor按行排列，即一行对应一个anchor属性，此时的属性仍然是feature map得到的值
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    # 锚点的维度与net块的height和width属性一致。这些属性描述了输入图像的维度，比feature map的规模大（二者之商即是步幅）。因此，我们必须使用stride分割锚点。变换后的anchors是相对于最终的feature map的尺寸
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the tX, tY. and object confidencce.tx与ty为预测的坐标偏移值
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #这里生成了每个格子的左上角坐标，生成的坐标为grid x grid的二维数组，a，b分别对应这个二维矩阵的x,y坐标的数组，a,b的维度与grid维度一样。每个grid cell的尺寸均为1，故grid范围是[0,12]（假如当前的特征图13*13）
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    #x_offset即cx,y_offset即cy，表示当前cell左上角坐标
    x_offset = torch.FloatTensor(a).view(-1,1)#view是reshape功能，-1表示自适应
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    #这里的x_y_offset对应的是最终的feature map中每个格子的左上角坐标，比如有13个格子，刚x_y_offset的坐标就对应为(0,0),(0,1)…(12,12) .view(-1, 2)将tensor变成两列，unsqueeze(0)在0维上添加了一维。
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset#bx=sigmoid(tx)+cx,by=sigmoid(ty)+cy

   
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    # 这里的anchors本来是一个长度为6的list(三个anchors每个2个坐标)，然后在0维上(行)进行了grid_size*grid_size个复制，在1维(列)上
    # 一次复制(没有变化)，即对每个格子都得到三个anchor。Unsqueeze(0)的作用是在数组上添加一维，这里是在第0维上添加的。添加grid_size是为了之后的公式bw=pw×e^tw的tw。
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    #对网络预测得到的矩形框的宽高的偏差值进行指数计算，然后乘以anchors里面对应的宽高(这里的anchors里面的宽高是对应最终的feature map尺寸grid_size)，
    # 得到目标的方框的宽高，这里得到的宽高是相对于在feature map的尺寸
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors#公式bw=pw×e^tw及bh=ph×e^th，pw为anchorbox的长度
    # 这里得到每个anchor中每个类别的得分。将网络预测的每个得分用sigmoid()函数计算得到
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride#将相对于最终feature map的方框坐标和尺寸映射回输入网络图片(416x416)，即将方框的坐标乘以网络的stride即可
    
    return prediction



    '''
    必须使我们的输出满足 objectness 分数阈值和非极大值抑制（NMS），以得到后文所提到的「真实」检测结果。要做到这一点就要用 write_results函数。
    函数的输入为预测结果、置信度（objectness 分数阈值）、num_classes（我们这里是 80）和 nms_conf（NMS IoU 阈值）。
    write_results()首先将网络输出方框属性(x,y,w,h)转换为在网络输入图片(416x416)坐标系中，方框左上角与右下角坐标(x1,y1,x2,y2)，以方便NMS操作。
    然后将方框含有目标得分低于阈值的方框去掉，提取得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score,
    然后进行NMS操作。最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls)，ind 是这个方框所属图片在这个batch中的序号，
    x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入图片(416x416)坐标系中，方框右下角的坐标。
    s是这个方框含有目标的得分,s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号.
    '''
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # confidence: 输入的预测shape=(1,10647, 85)。conf_mask: shape=(1,10647) => 增加一维度之后 (1, 10647, 1)
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)#我们的预测张量包含有关Bx10647边界框的信息。对于含有目标的得分小于confidence的每个方框，它对应的含有目标的得分将变成0,即conf_mask中对应元素为0.而保留预测结果中置信度大于给定阈值的部分prediction的conf_mask
    prediction = prediction*conf_mask # 小于置信度的条目值全为0, 剩下部分不变。conf_mask中含有目标的得分小于confidence的方框所对应的含有目标的得分为0，
    #根据numpy的广播原理，它会扩展成与prediction维度一样的tensor，所以含有目标的得分小于confidence的方框所有的属性都会变为0，故如果没有检测任何有效目标,则返回值为0
   
    '''
    保留预测结果中置信度大于阈值的bbox
    下面开始为nms准备
    '''

    # prediction的前五个数据分别表示 (Cx, Cy, w, h, score)，这里创建一个新的数组，大小与predicton的大小相同   
    box_corner = prediction.new(prediction.shape)
    '''
    我们可以将我们的框的 (中心 x, 中心 y, 高度, 宽度) 属性转换成 (左上角 x, 左上角 y, 右下角 x, 右下角 y)
    这样做用每个框的两个对角坐标能更轻松地计算两个框的 IoU
    '''
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)# x1 = Cx - w/2
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)# y1 = Cy - h/2
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)# x2 = Cx + w/2 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)# y2 = Cy + h/2
    prediction[:,:,:4] = box_corner[:,:,:4]# 计算后的新坐标复制回去
    
    batch_size = prediction.size(0)# 第0个维度是batch_size
    # output = prediction.new(1, prediction.size(2)+1) # shape=(1,85+1)
    write = False # 拼接结果到output中最后返回
    
    #对每一张图片得分的预测值进行NMS操作，因为每张图片的目标数量不一样，所以有效得分的方框的数量不一样，没法将几张图片同时处理，因此一次只能完成一张图的置信度阈值的设置和NMS,不能将所涉及的操作向量化.
    #所以必须在预测的第一个维度上（batch数量）上遍历每张图片，将得分低于一定分数的去掉，对剩下的方框进行进行NMS
    for ind in range(batch_size):
        image_pred = prediction[ind]          # 选择此batch中第ind个图像的预测结果,image_pred对应一张图片中所有方框的坐标(x1,y1,x2,y2)以及得分，是一个二维tensor 维度为10647x85
      
        # 最大值索引, 最大值, 按照dim=1 方向计算
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)#我们只关心有最大值的类别分数，prediction[:, 5:]表示每一分类的分数,返回每一行中所有类别的得分最高的那个类的得分max_conf，同时返回这个类对应的序号max_conf_score
        # 维度扩展max_conf: shape=(10647->15) => (10647->15,1)添加一个列的维度，max_conf变成二维tensor，尺寸为10647x1
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)#我们移除了每一行的这 80 个类别分数，只保留bbox4个坐标以及objectnness分数，转而增加了有最大值的类别分数及索引。
        #将每个方框的(x1,y1,x2,y2,s)与得分最高的这个类的分数s_cls(max_conf)和对应类的序号index_cls(max_conf_score)在列维度上连接起来，
        # 即将10647x5,10647x1,10647x1三个tensor 在列维度进行concatenate操作，得到一个10647x7的tensor,(x1,y1,x2,y2,s,s_cls,index_cls)。
        image_pred = torch.cat(seq, 1)# shape=(10647, 5+1+1=7)
        #image_pred[:,4]是长度为10647的一维tensor,维度为4的列是置信度分数。假设有15个框含有目标的得分非0，返回15x1的tensor
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))#torch.nonzero返回的是索引，会让non_zero_ind是个2维tensor
        try:# try-except模块的目的是处理无检测结果的情况.non_zero_ind.squeeze()将15x1的non_zero_ind去掉维度为1的维度，变成长度为15的一维tensor，相当于一个列向量，
            # image_pred[non_zero_ind.squeeze(),:]是在image_pred中找到non_zero_ind中非0目标得分的行的所有元素(image_pred维度
            # 是10647x7，找到其中的15行)， 再用view(-1,7)将它变为15x7的tensor，用view()确保第二个维度必须是7.
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:#当没有检测到时目标时，我们使用 continue 来跳过对本图像的循环，即进行下一次循环。
            continue             
  
         # 获取当前图像检测结果中出现的所有类别
        img_classes = unique(image_pred_[:,-1])  #pred_[:,-1]是一个15x7的tensor，最后一列保存的是每个框里面物体的类别，-1表示取最后一列。
        #用unique()除去重复的元素，即一类只留下一个元素，假设这里最后只剩下了3个元素，即只有3类物体。
        
        #按照类别执行 NMS
        
        for cls in img_classes:
            #一旦我们进入循环，我们要做的第一件事就是提取特定类别（用变量 cls 表示）的检测结果,分离检测结果中属于当前类的数据 -1: index_cls, -2: s_cls
            '''
            本句是将image_pred_中属于cls类的预测值保持不变，其余的全部变成0。image_pred_[:,-1] == cls，返回一个与image_pred_
            行数一样的一维tensor，这里长度为15.当image_pred_中的最后一个元素(物体类别索引)等于第cls类时，返回的tensor对应元素为1，
            否则为0. 它与image_pred_相乘时，先扩展为15x7的tensor(似乎这里还没有变成15x7的tensor)，为0元素一行全部为0，再与
            image_pred_相乘，属于cls这类的方框对应预测元素不变，其它类的为0.unsqueeze(1)添加了列这一维，变成15x7的二维tensor。
            '''
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1) 
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze() #cls_mask[:,-2]为cls_mask倒数第二列,是物体类别分数。
#cls_mask本身为15x7，cls_mask[:,-2]将cls_mask的倒数第二列取出来，此时是1维tensor，torch.nonzero(cls_mask[:,-2])得到的是非零元素的索引，
#将返回一个二维tensor，这里是4x2，再用squeeze()去掉长度为1的维度(这里是第二维)，得到一维tensor(相当于一列)。
            image_pred_class = image_pred_[class_mask_ind].view(-1,7) #从prediction中取出属于cls类别的所有结果，为下一步的nms的输入.
#找到image_pred_中对应cls类的所有方框的预测值，并转换为二维张量。这里4x7。image_pred_[class_mask_ind]本身得到的数据就是4x7，view(-1,7)是为了确保第二维为7
            ''' 到此步 prediction_class 已经存在了我们需要进行非极大值抑制的数据 '''
            # 开始 nms
            # 按照score排序, 由大到小
            # 最大值最上面            
            
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]# # 这里的sort()将返回两个tensor，第一个是每个框含有有目标的分数由低到高排列，第二个是现在由高到底的tensor中每个元素在原来的序号。[0]是排序结果, [1]是排序结果的索引
            image_pred_class = image_pred_class[conf_sort_index]#根据排序后的索引对应出的bbox的坐标与分数，依然为4x7的tensor
            idx = image_pred_class.size(0)   #detections的个数

            '''开始执行 "非极大值抑制" 操作'''
            for i in range(idx):
                # 对已经有序的结果，每次开始更新后索引加一，挨个与后面的结果比较
                
                try:# image_pred_class[i].unsqueeze(0)，为什么要加unsqueeze(0)？这里image_pred_class为4x7的tensor，image_pred_class[i]是一个长度为7的tensor，要变成1x7的tensor，在第0维添加一个维度。
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])#这句话的作用是计算第i个方框和i+1到最终的所有方框的IOU。
                except ValueError:
                '''
                在for i in range(idx):这个循环中，因为有一些框(在image_pred_class对应一行)会被去掉，image_pred_class行数会减少，
                这样在后面的循环中，idx序号会超出image_pred_class的行数的范围，出现ValueError错误。
                所以当抛出这个错误时，则跳出这个循环，因为此时已经没有更多可以去掉的方框了。
                '''
                    break
            
                except IndexError:
                    break
            
                
                iou_mask = (ious < nms_conf).float().unsqueeze(1)# 计算出需要保留的item（保留ious < nms_conf的框）而ious < nms_conf得到的是torch.uint8类型，用float()将它们转换为float类型。因为要与image_pred_class[i+1:]相乘，故长度为7的tensor，要变成1x7的tensor，需添加一个维度。
                image_pred_class[i+1:] *= iou_mask #将iou_mask与比序号i大的框的预测值相乘，其中IOU大于阈值的框的预测值全部变成0.得出需要保留的框    
            
                # 开始移除
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()#torch.nonzero返回的是索引，是2维tensor。将经过iou_mask掩码后的每个方框含有目标的得分为非0的方框的索引提取出来，non_zero_ind经squeeze后为一维tensor，含有目标的得分非0的索引
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)#得到含有目标的得分非0的方框的预测值(x1, y1, x2, y2, s,  s_class,index_cls)，为1x7的tensor
            # 当前类的nms执行完之后，下一次循环将对剩下的方框中得分第i+1高的方框进行NMS操作，因为刚刚已经对得分第1到i高的方框进行了NMS操作。直到最后一个方框循环完成为止
            # 在每次进行NMS操作的时候，预测值tensor中都会有一些行(对应某些方框)被去掉。接下来是保存结果。
            # new()创建了一个和image_pred_class类型相同的tensor，tensor行数等于cls这个类别所有的方框经过NMS剩下的方框的个数，即image_pred_class的行数，列数为1.
            #再将生成的这个tensor所有元素赋值为这些方框所属图片对应于batch中的序号ind(一个batch有多张图片同时测试)，用fill_(ind)实现   
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)     
            seq = batch_ind, image_pred_class
            #我们没有初始化我们的输出张量，除非我们有要分配给它的检测结果。一旦其被初始化，我们就将后续的检测结果与它连接起来。我们使用write标签来表示张量是否初始化了。在类别上迭代的循环结束时，我们将所得到的检测结果加入到张量输出中。
            if not write:
            # 将batch_ind, image_pred_class在列维度上进行连接，image_pred_class每一行存储的是(x1,y1,x2,y2,s,s_cls,index_cls)，现在在第一列增加了一个代表这个行对应方框所属图片在一个batch中的序号ind
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    try:#在该函数结束时，我们会检查输出是否已被初始化。如果没有，就意味着在该 batch 的任意图像中都没有单个检测结果。在这种情况下，我们返回 0。
        return output
    except:# 如果所有的图片都没有检测到方框，则在前面不会进行NMS等操作，不会生成output，此时将在except中返回0
        return 0
    # 最终返回的output是一个batch中所有图片中剩下的方框的属性，一行对应一个方框，属性为(x1,y1,x2,y2,s,s_cls,index_cls)，
    # ind 是这个方框所属图片在这个batch中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是在网络输入
    # 图片(416x416)坐标系中，方框右下角的坐标。s是这个方框含有目标的得分s_cls是这个方框中所含目标最有可能的类别的概率得分，index_cls是s_cls对应的这个类别所对应的序号

def letterbox_image(img, inp_dim):
    """
    lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充,调整图像尺寸
    具体而言,此时某个边正好可以等于目标长度,另一边小于等于目标长度
    将缩放后的数据拷贝到画布中心,返回完成缩放
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim　#inp_dim是需要resize的尺寸（如416*416）
    # 取min(w/img_w, h/img_h)这个比例来缩放，缩放后的尺寸为new_w, new_h,即保证较长的边缩放后正好等于目标长度(需要的尺寸)，另一边的尺寸缩放后还没有填充满.
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC) #将图片按照纵横比不变来缩放为new_w x new_h，768 x 576的图片缩放成416x312.,用了双三次插值
    # 创建一个画布, 将resized_image数据拷贝到画布中心。
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)#生成一个我们最终需要的图片尺寸hxwx3的array,这里生成416x416x3的array,每个元素值为128
    # 将wxhx3的array中对应new_wxnew_hx3的部分(这两个部分的中心应该对齐)赋值为刚刚由原图缩放得到的数组,得到最终缩放后图片
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):#prep_image用来将numpy数组转换成PyTorch需要的的输入格式。即（3，416,416）
    """
    为神经网络准备输入图像数据
    返回值: 处理后图像, 原图, 原图尺寸
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))# lteerbox_image()将图片按照纵横比进行缩放，将空白部分用(128,128,128)填充
    img = img[:,:,::-1].transpose((2,0,1)).copy()#img是【h,w,channel】，这里的img[:,:,::-1]是将第三个维度channel从opencv的BGR转化为pytorch的RGB，然后transpose((2,0,1))的意思是将[height,width,channel]->[channel,height,width]
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)# from_numpy(()将ndarray数据转换为tensor格式，div(255.0)将每个元素除以255.0，进行归一化，unsqueeze(0)在0维上添加了一维，
# 从3x416x416变成1x3x416x416，多出来的一维表示batch。这里就将图片变成了BxCxHxW的pytorch格式
    return img

def load_classes(namesfile): #load_classes会返回一个字典——将每个类别的索引映射到其名称的字符串
    """
    加载类名文件
    :param namesfile:
    :return: 元组,包括类名数组和总类的个数
    """
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
```

