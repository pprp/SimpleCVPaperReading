"""-*-coding:utf-8-*-
Test the cub validation result
"""
import os
import csv
import json
import random
import torch
import torch.nn as nn
import numpy as np
import horovod.torch as hvd
import urllib.request as urt
import torch.backends.cudnn as cudnn

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from scipy.special import softmax
from torchvision.transforms import transforms
from models.build_model import BuildModel
from torch.utils.data import Dataset, DataLoader


PATH = "/data/remote/yy_git_code/cub_baseline"


class TestDataSet(Dataset):
    def __init__(self):
        super(TestDataSet, self).__init__()
        self.test_file = "/data/remote/yy_git_code/cub_baseline/dataset/test_accv.txt"
        # self.test_file = "/data/remote/yy_git_code/cub_baseline/dataset/train_accv_pingtai_clean_gif.txt"
        # self.test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
        self.test_list = [(x.strip().split(',')[0], int(
            float(x.strip().split(',')[1]))) for x in open(self.test_file).readlines()]
        self.Resize_size = 299
        # self.input_size = 300
        self.imagenet_normalization_paramters = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        url, lbl = self.test_list[idx][0], self.test_list[idx][1]
        image = Image.open(BytesIO(urt.urlopen(url).read()))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = transforms.Resize(
            (self.Resize_size, self.Resize_size), Image.BILINEAR)(image)
        image = transforms.ToTensor()(image)
        image = self.imagenet_normalization_paramters(image)
        return image, url

    def __len__(self):
        return len(self.test_list)


class ModelFeature(nn.Module):
    def __init__(self, net):
        super(ModelFeature, self).__init__()
        self.net = net

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x


def AccvModel(model_name, num_classes, model_weights):
    is_pretrained = False
    net = BuildModel(model_name, num_classes, is_pretrained)()
    if model_weights == "" or model_weights is None:
        return net
    else:
        model_state_dict = torch.load(model_weights, map_location="cpu")
        net.load_state_dict(model_state_dict["model"])
        print("Load the accv dataset model")
        return net


def infer_batch(net, data):
    net.eval()
    data = data.cuda()
    with torch.no_grad():
        logits = net(data)
    return logits


def infer_image(net, data):
    net.eval()
    data = data.cuda()
    with torch.no_grad():
        logits = model(data).cpu().numpy()
        print(logits.shape)
        for i in range(logits.shape[0]):
            prob = softmax(logits[i])
            lbl = np.argmax(prob)
            print(lbl)
        # for i in range(lbl.shape[0]):
        #     print(i)


def calculate_accuracy(test_gt, test_pd):
    gt_dict = {x.strip().split(',')[0]: x.strip().split(',')[
        1] for x in open(test_gt).readlines()}
    pd_dict = {x.strip().split(',')[0]: x.strip().split(',')[
        1] for x in open(test_pd).readlines()}
    count = 0
    for key, value in pd_dict.items():
        if key in gt_dict.keys():
            if value == gt_dict[key]:
                count += 1
    print("Accuracy: ", count / len(pd_dict))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 避免产生结果的随机性
    torch.backends.cudnn.deterministic = True

# predict_logits


def predict_logits(model, data_loader):
    model = model.cuda()
    model = model.eval()
    result_logits = []
    with tqdm(total=len(data_loader), desc="processing predict logits", disable=not verbose) as t:
        with torch.no_grad():
            for idx, data in tqdm(enumerate(data_loader)):
                image_tensor, image_path = data[0], data[1]
                data_logits = infer_batch(model, image_tensor).cpu().numpy()
                for i in range(len(image_path)):
                    result = {"image_path": image_path[i].split(
                        '/')[-1], "image_logits": data_logits[i].tolist()}
                    result_logits.append(result)
    return result_logits


# test model
def test_logits(model, data_loader):
    model = model.cuda()
    model = model.eval()
    for idx, data in enumerate(data_loader):
        image_tensor, image_path = data[0], data[1]
        infer_image(model, image_tensor)
        break


if __name__ == "__main__":

    setup_seed(42)
    hvd.init()

    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    cudnn.benchmark = True

    test_file = "/data/remote/yy_git_code/cub_baseline/dataset/test_accv.txt"
    # test_file = "/data/remote/yy_git_code/cub_baseline/dataset/train_accv_pingtai_clean_gif.txt"
    # test_file = "/data/remote/code/classification_trick_with_model/data/val_imagenet_128w.txt"
    test_dict = {x.split(',')[0]: int(float(x.split(',')[1]))
                 for x in open(test_file).readlines()}

    batch_size = 32
    num_workers = 32
    num_classes = 5000

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if torch.cuda.is_available() else {}

    test_dataset = TestDataSet()
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
    )
    testLoader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler, **kwargs)
    verbose = 1 if hvd.rank() == 0 else 0

    # model_ckpt = "/data/remote/output_ckpt_with_logs/accv/ckpt/regnet-clean-224/checkpoint-epoch-38.pth.tar"
    model_ckpt = "/data/remote/output_ckpt_with_logs/accv/ckpt/inceptionv3-pretrain-clean-299/checkpoint-epoch-88.pth.tar"
    # model_ckpt = "/data/remote/output_ckpt_with_logs/accv/ckpt/r50_448_48.312.pth.tar"
    # model_ckpt = None
    # model = CUBModel(model_name="resnet50", num_classes=5000, model_weights=model_ckpt)
    model = AccvModel(model_name="inceptionv3",
                      num_classes=num_classes, model_weights=model_ckpt)
    # print(model)
    # model_feature = ModelFeature(model)

    # use hvd ddp
    logits_result = predict_logits(model, testLoader)
    for i in range(hvd.size()):
        if hvd.rank() == i:
            np.save("/data/remote/output_ckpt_with_logs/accv/logits/inceptionv3/incepv3_{}.npy".format(i),np.array(logits_result))

    # # test_logits(model, testLoader)
