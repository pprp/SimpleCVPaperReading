---
title: pytorch-network-slimming代码LIST
date: 2019-11-25 20:39:26
tags:
- pytorch
- networkslimming
---

# Network Slimming

train.py:

```python
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import cifar

from networks import resnet18, vgg11, vgg11s, densenet63
from netslim import prune, load_pruned_model, update_bn, update_bn_by_names, get_norm_layer_names, liu2017_normalized_by_layer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

archs = {
    "resnet18": resnet18, 
    "vgg11": vgg11, "vgg11s": vgg11s, 
    "densenet63": densenet63
}

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar-100 Example for Network Slimming')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--resume-path', default='',
                        help='path to a trained model weight')
    parser.add_argument('--arch', default='resnet18',
                        help='network architecture')
    parser.add_argument('--epochs', type=int, default=220, metavar='EP',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-decay-epochs', type=int, default=50, metavar='LR-T',
                        help='the period of epochs to decay LR')
    parser.add_argument('--lr-decay-factor', type=float, default=0.3162, metavar='LR-MUL',
                        help='decay factor of learning rate (default: 0.3162)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='L2',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--l1-decay', type=float, default=0, metavar='L1',
                        help='coefficient for L1 regularization on BN (default: 0)')
    parser.add_argument('--prune-ratio', type=float, default=-1, metavar='PR',
                        help='ratio of pruned channels to total channels, -1: do not prune')
    parser.add_argument('--all-bn', action='store_true', default=False,
                        help='L1 regularization on all BNs, otherwise only on prunable BN')
    parser.add_argument('--momentum', type=float, default=0.85, metavar='M',
                        help='SGD momentum (default: 0.85)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='LOG-T',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--outf', default='output-cifar-100', metavar='OUTNAME', 
                        help='folder to output images and model checkpoints')
    parser.add_argument('--tfs', action='store_true', default=False,
                        help='train from scratch')
    parser.add_argument('--experimental', action='store_true', default=False,
                        help='Normalize scaling factor per layer for pruning')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if args.cuda else 'cpu')

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_loader = torch.utils.data.DataLoader(
        cifar.CIFAR100('./cifar-100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           normalize
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        cifar.CIFAR100('./cifar-100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = archs[args.arch](num_classes=100)
    bn_names = []
    if args.l1_decay > 0:
        if not args.all_bn:
            bn_names = get_norm_layer_names(model, (3, 32, 32))
            print("Sparsity regularization will be applied to:")
            for bn_name in bn_names:
                print(bn_name)
        else:
            print("Sparsity regularization will be applied to all BN layers:")

    model = model.to(device)

    if args.resume_path:
        try:
            model.load_state_dict(torch.load(args.resume_path))
        except:
            print("Failed to load state_dict directly, trying to load pruned weight ...")
            model = load_pruned_model(model, torch.load(args.resume_path))
        if args.prune_ratio > 0:
            if args.experimental:
                model = prune(model, (3, 32, 32), args.prune_ratio, prune_method=liu2017_normalized_by_layer)
            else:
                model = prune(model, (3, 32, 32), args.prune_ratio)
        if args.tfs:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if args.arch == "densenet63":
                        nn.init.kaiming_normal_(m.weight)  # weird??
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    m.reset_parameters()
                    if args.arch == "densenet63":
                        m.reset_parameters()
                        nn.init.constant_(m.bias, 0)
                    elif args.arch in ["vgg11", "vgg11s"]:
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)             
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            print("Train pruned model from scratch ...")

    lsm = nn.LogSoftmax(dim=1)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        nesterov=True, 
        weight_decay=args.weight_decay
    )

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(lsm(output), target)
            loss.backward()
            if args.all_bn:
                update_bn(model, args.l1_decay)
            else:
                update_bn_by_names(model, bn_names, args.l1_decay)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(lsm(output), target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1]   # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * float(correct) / float(len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))
        with open('{}.log'.format(args.outf), 'a') as f:
            f.write('{}\t{}\n'.format(epoch, accuracy))
        return accuracy

    max_accuracy = 0.
    os.system('mkdir -p {}'.format(args.outf))

    lr = args.lr
    for epoch in range(args.epochs):
        if epoch > 0 and epoch % args.lr_decay_epochs == 0:
            lr *= args.lr_decay_factor
            print('Changing learning rate to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train(epoch)
        accuracy = test(epoch)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), '{}/ckpt_best.pth'.format(args.outf))
        torch.save(model.state_dict(), '{}/ckpt_last.pth'.format(args.outf))
```

test.py

```python
import os
import argparse
import time
import torch
from torchvision import transforms
from torchvision.datasets import cifar
from networks import resnet18, vgg11, vgg11s, densenet63
from netslim import load_pruned_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

archs = {
    "resnet18": resnet18, 
    "vgg11": vgg11, "vgg11s": vgg11s, 
    "densenet63": densenet63
}

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar-100 Example for Test Pruned Model')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 50)')
parser.add_argument('--resume-path', default='',
                    help='path to a trained model weight')
parser.add_argument('--arch', default='resnet18',
                    help='network architecture')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

test_loader = torch.utils.data.DataLoader(
    cifar.CIFAR100('./cifar-100', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = archs[args.arch](num_classes=100)
pruned_weights = torch.load(args.resume_path)
model = load_pruned_model(model, pruned_weights).to(device)

model.eval()
correct = 0
with torch.no_grad():
    t_start = time.time()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]   # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    t_all = time.time() - t_start

accuracy = 100. * float(correct) / float(len(test_loader.dataset))
print("Accuracy: {}/{} ({:.2f}%)\n".format(correct, len(test_loader.dataset), accuracy))
print("Total time: {:.2f} s".format(t_all))
#if args.test_batch_size == 1:
#    print("Estimated FPS: {:.2f}".format(1/(t_all/len(test_loader))))
```

network.py

```python
import torch
import torch.nn as nn
import torchvision.models as models

def resnet18(num_classes=100):
    """Constructs a ResNet-18 model for CIFAR dataset"""
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.avgpool = nn.AvgPool2d(4, stride=1)
    model.maxpool = nn.Identity()
    return model

def vgg11(num_classes=100):
    """Constructs a VGG-11 model for CIFAR dataset"""
    model = models.vgg11_bn(num_classes=num_classes)
    model.avgpool = nn.Identity()
    model.classifier[0] = nn.Linear(512, 4096)
    return model

def vgg11s(num_classes=100):
    """Constructs a VGG-11 simplified model for CIFAR dataset"""
    model = models.vgg11_bn()
    model.avgpool = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    return model

def vgg11mk2(num_classes=100):
    """Constructs a VGG-11 BN classifier model for CIFAR dataset"""
    model = models.vgg11_bn(num_classes=num_classes)
    model.avgpool = nn.Identity()
    model.classifier[0] = nn.Linear(512, 4096)
    model.classifier[2] = nn.BatchNorm1d(4096)
    model.classifier[5] = nn.BatchNorm1d(4096)
    return model

def densenet63(num_classes=100):
    """Constructs a DenseNet-63 simplified model for CIFAR dataset"""
    num_init_features = 32
    model = models.densenet._densenet('densenet63', 32, (3, 6, 12, 8), num_init_features, pretrained=False, progress=False)
    model.features[0] = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[1] = nn.BatchNorm2d(num_init_features)
    model.features[3] = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    return model

if __name__ == "__main__":
    from torchsummary import summary
    model = densenet63(num_classes=100)
    summary(model, (3, 32, 32), device="cpu")
    print(model)
```

statistics.py

```python
import sys
import torch
from netslim import load_pruned_model
from thop import profile

from networks import resnet18, vgg11, vgg11s, densenet63

archs = {
    "resnet18": resnet18, 
    "vgg11": vgg11, "vgg11s": vgg11s, 
    "densenet63": densenet63
}

arch_name = sys.argv[1]
weight_path = sys.argv[2]
model = archs[arch_name](num_classes=100)
weight = torch.load(weight_path, map_location="cpu")

try:
    model.load_state_dict(weight)
except:
    model = load_pruned_model(model, weight)
    
input_t = torch.randn(1, 3, 32, 32)
flops, params = profile(model, inputs=(input_t,), verbose=False)
flops_str = format(int(flops), ',')
gflops = flops / 1024**3
gflops_str = "{:.2f} GFLOPS".format(gflops)
params_str = format(int(params), ',')
mparams = params / 1024**2
mparams_str = "{:.2f} M".format(mparams)
line = "{}/{}: FLOPS: {} / {}\t# of params: {} / {}".format(arch_name, weight_path, flops_str, gflops_str, params_str, mparams_str)
print(line)
```

netslim/graph_parser.py

```python
import re
import torch

CHANNEL_DIM = 1
NORM_LAYER_KEYWORDS = ["batch_norm", "group_norm", "instance_norm"]
PRUNABLE_LAYER_KEYWORDS = ["convolution", "addmm"]  # does not support groups > 1 for conv
PASS_KEYWORDS = ["relu", "leaky_relu", "sigmoid", "tanh",
                 "pool", "pad", "dropout",
                 "view", ]  # and more .. does not support concat
OTHER_OP_KEYWORDS = ["cat"]
OTHER_PRIM_KEYWORDS = ["ListConstruct"]  # for cat
NO_EFFECT_KEYWORDS = ["size", ]

scope_pattern = re.compile(r".+, scope: (.+)")
module_pattern = re.compile(r"\[(\w+)\]")
output_pattern = re.compile(r"(%.+) : .*, scope: .+")
input_pattern = re.compile(r".+ = aten::\w+\((%.+),*\), scope: .+")
prim_input_pattern = re.compile(r".+ = prim::\w+\((%.+),*\), scope: .+")
shape_pattern = re.compile(r"%.+ : \w+\((.+)\) = aten::\w+\(%.+\), scope: .+")
int_pattern = re.compile(r"[1-9]+")
view_pattern = re.compile(r"aten::.*view.*")

norm_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in NORM_LAYER_KEYWORDS])
    )
)
prunable_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in PRUNABLE_LAYER_KEYWORDS])
    )
)
pass_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in PASS_KEYWORDS])
    )
)
allowed_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in
                  PRUNABLE_LAYER_KEYWORDS +
                  PASS_KEYWORDS +
                  NO_EFFECT_KEYWORDS])
    )
)
common_layer_pattern = re.compile(
    r".*= ({})\(.*, scope: .+".format(
        '|'.join(["aten::\w*{}\w*".format(_) for _ in
                  NORM_LAYER_KEYWORDS +
                  PRUNABLE_LAYER_KEYWORDS +
                  PASS_KEYWORDS +
                  OTHER_OP_KEYWORDS]
                 +["prim::\w*{}\w*".format(_) for _ in
                   OTHER_PRIM_KEYWORDS])
    )
)
tensor_op_pattern = re.compile(r".*= (aten::\w+)\(.*, scope: .+")


def get_node_str(node):
    return repr(node).split(" # ")[0]


def parse_module_name(x):
    scope_found = scope_pattern.findall(x)
    module_name = ''
    if scope_found:
        tokens = scope_found[0].split('/')[1:]
        module_name = '.'.join([module_pattern.findall(_)[0] for _ in tokens])
    return module_name


def parse_output_name(x):
    return output_pattern.findall(x)[0]


def parse_input_names(x):
    result = input_pattern.findall(x)
    if not result:
        result = prim_input_pattern.findall(x)
    return result[0].split(", ")


def parse_output_shape(x):
    sizes = shape_pattern.findall(x)[0].split(", ")
    for s in sizes:
        if not int_pattern.match(s):
            return None
    return [int(_) for _ in sizes]


# assume for a normalization layer, it has only one input/output
def get_norm_layer_io(graph):
    out2nl = {}
    in2nl = {}
    for node in graph.nodes():
        node_str = get_node_str(node)
        if norm_layer_pattern.match(node_str):
            bn_name = parse_module_name(node_str)
            output = parse_output_name(node_str)
            input = parse_input_names(node_str)[0]
            out2nl[output] = bn_name
            in2nl[input] = bn_name
    return out2nl, in2nl


def reverse_search_dict(val, target_dict):
    return [k for k, v in target_dict.items() if v == val]


# check for tensor operation layer and prim::ListConstruct, which is used by cat operation
def get_input_count(graph):
    input_count = {}
    for node in graph.nodes():
        node_str = get_node_str(node)
        matches = common_layer_pattern.findall(node_str)
        if matches:
            input_names = parse_input_names(node_str)
            for input_name in input_names:
                if input_name not in input_count:
                    input_count[input_name] = 1
                else:
                    input_count[input_name] += 1
    return input_count


def get_pruning_layers(model, input_shape):
    """parse the model graph, and generate mapping to BNs
    Arguments:
        model (pytorch model): the model instance
        input_shape (tuple): shape of the input tensor
    Returns:
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
    """

    # 0 trace graph with torch scripts
    inputs = torch.randn(2, *input_shape)   # 2 for BatchNorm1d
    trace, _ = torch.jit.get_trace_graph(model, args=(inputs,))
    graph = trace.graph()
    input_count = get_input_count(graph)

    # 1 get norm layers and their direct outputs/inputs
    output2norm, input2norm = get_norm_layer_io(graph)

    # 2 find & update all possible outputs/inputs that with per-channel operations only
    # assume for a per-channel operation layer, it has only one input/output
    new_outputs = list(output2norm.keys())
    tensor_shape = {}
    while new_outputs:
        temp_outputs = new_outputs[:]
        for node in graph.nodes():
            node_str = get_node_str(node)
            # found new outputs
            matches = pass_layer_pattern.findall(node_str)
            if matches:
                input_names = parse_input_names(node_str)
                for input_name in input_names:
                    if input_name in temp_outputs:
                        output_name = parse_output_name(node_str)
                        if output_name not in tensor_shape:
                            output_shape = parse_output_shape(node_str)
                            tensor_shape[output_name] = output_shape

                        # check channel dim consistency for view operation
                        if view_pattern.match(matches[0]):
                            if tensor_shape[output_name][CHANNEL_DIM] == tensor_shape[input_name][CHANNEL_DIM]:
                                output2norm[output_name] = output2norm[input_name]
                                new_outputs.append(output_name)
                        # process normally
                        else:
                            output2norm[output_name] = output2norm[input_name]
                            new_outputs.append(output_name)
        new_outputs = new_outputs[len(temp_outputs):]

    new_inputs = list(input2norm.keys())
    while new_inputs:
        temp_inputs = new_inputs[:]
        for node in graph.nodes():
            node_str = get_node_str(node)
            # found new inputs
            matches = pass_layer_pattern.findall(node_str)
            if matches:
                output_name = parse_output_name(node_str)
                if output_name not in tensor_shape:
                    output_shape = parse_output_shape(node_str)
                    tensor_shape[output_name] = output_shape

                if output_name in temp_inputs:
                    input_name = parse_input_names(node_str)[0]

                    # check channel dim consistency for view operation
                    if view_pattern.match(matches[0]):
                        if tensor_shape[output_name][CHANNEL_DIM] == tensor_shape[input_name][CHANNEL_DIM]:
                            input2norm[input_name] = input2norm[output_name]
                            new_inputs.append(input_name)
                    # process normally
                    else:
                        input2norm[input_name] = input2norm[output_name]
                        new_inputs.append(input_name)
        new_inputs = new_inputs[len(temp_inputs):]

    # 3 identify layers need to be pruned
    succ_layers = {}    # succeeding layers
    prec_layers = {}    # preceding layers
    risky_layer_names = []
    for node in graph.nodes():
        node_str = get_node_str(node)
        if tensor_op_pattern.match(node_str):
            input_names = parse_input_names(node_str)
            output_name = parse_output_name(node_str)
            for input_name in input_names:
                if input_name in output2norm:
                    layer_name = parse_module_name(node_str)
                    source_layer_name = output2norm[input_name]
                    if prunable_layer_pattern.match(node_str):
                        # normalized output may be inputs to multiple layers
                        if source_layer_name in succ_layers:
                            succ_layers[source_layer_name].append(layer_name)
                        else:
                            succ_layers[source_layer_name] = [layer_name, ]
                    if not allowed_layer_pattern.match(node_str):
                        risky_layer_names.append(source_layer_name)

            if output_name in input2norm:
                layer_name = parse_module_name(node_str)
                source_layer_name = input2norm[output_name]

                if prunable_layer_pattern.match(node_str):
                    # support single input to normalization layer
                    prec_layers[source_layer_name] = [layer_name, ]
                if not allowed_layer_pattern.match(node_str):
                    # check for not allowed layers
                    risky_layer_names.append(source_layer_name)

                # make sure there are no branches in the path
                norm_inputs = reverse_search_dict(source_layer_name, input2norm)
                for norm_input in norm_inputs:
                    if input_count[norm_input] > 1:
                        risky_layer_names.append(source_layer_name)
                        break

    risky_layer_names = list(set(risky_layer_names))
    for risky_layer_name in risky_layer_names:
        if risky_layer_name in succ_layers:
            succ_layers.pop(risky_layer_name)
        if risky_layer_name in prec_layers:
            prec_layers.pop(risky_layer_name)

    return prec_layers, succ_layers

def get_norm_layer_names(model, input_shape):
    prec_layers, succ_layers = get_pruning_layers(model, input_shape)
    return list(set(succ_layers) & set(prec_layers))
```

netslim/prune.py

```python
import copy
from functools import partial
import torch
import torch.nn as nn
from .graph_parser import get_pruning_layers

OUT_CHANNEL_DIM = 0
IN_CHANNEL_DIM = 1
WEIGHT_POSTFIX = ".weight"
BIAS_POSTFIX = ".bias"
MIN_CHANNELS = 3


def group_weight_names(weight_names):
    grouped_names = {}
    for weight_name in weight_names:
        group_name = '.'.join(weight_name.split('.')[:-1])
        if group_name not in grouped_names:
            grouped_names[group_name] = [weight_name, ]
        else:
            grouped_names[group_name].append(weight_name)
    return grouped_names


def liu2017(weights, prune_ratio, prec_layers, succ_layers, per_layer_normalization=False):
    """default pruning method as described in:
            Zhuang Liu et.al., "Learning Efficient Convolutional Networks through Network Slimming", in ICCV 2017"
    Arguments:
        weights (OrderedDict): unpruned model weights
        prune_ratio (float): ratio of be pruned channels to total channels
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
        per_layer_normalization (bool): if do normalization by layer
    Returns:
        pruned_weights (OrderedDict): pruned model weights
    """

    # find all scale weights in BN layers
    scale_weights = []
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        weight = weights[norm_weight_name]
        if per_layer_normalization:
            scale_weights.extend([(_.abs()/weight.sum()).item() for _ in list(weight)])
        else:
            scale_weights.extend([_.abs().item() for _ in list(weight)])

    # find threshold for pruning
    scale_weights.sort()
    prune_th_index = int(float(len(scale_weights)) * prune_ratio + 0.5)
    prune_th = scale_weights[prune_th_index]

    # unpruned_norm_layer_names = list(set(succ_layers) ^ set(prec_layers))
    grouped_weight_names = group_weight_names(weights.keys())
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        scale_weight = weights[norm_weight_name].abs()
        if per_layer_normalization:
            scale_weight = scale_weight / scale_weight.sum()
        prune_mask = scale_weight > prune_th
        if prune_mask.sum().item() == scale_weight.size(0):
            continue

        # in case not to prune the whole layer
        if prune_mask.sum() < MIN_CHANNELS:
            scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
            scale_weight_list.sort(reverse=True)
            prune_mask = scale_weight >= scale_weight_list[MIN_CHANNELS-1]

        prune_indices = torch.nonzero(prune_mask).flatten()
        
        # 1. prune source normalization layer
        for weight_name in grouped_weight_names[norm_layer_name]:
            weights[weight_name] = weights[weight_name].masked_select(prune_mask)

        # 2. prune target succeeding conv/linear/... layers
        for prune_layer_name in succ_layers[norm_layer_name]:
            for weight_name in grouped_weight_names[prune_layer_name]:
                if weight_name.endswith(WEIGHT_POSTFIX):
                    weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, prune_indices)

        # 3. prune target preceding conv/linear/... layers
        for prune_layer_name in prec_layers[norm_layer_name]:
            for weight_name in grouped_weight_names[prune_layer_name]:
                if weight_name.endswith(WEIGHT_POSTFIX):
                    weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, prune_indices)
                elif weight_name.endswith(BIAS_POSTFIX):
                    weights[weight_name] = weights[weight_name].index_select(0, prune_indices)

    return weights


liu2017_normalized_by_layer = partial(liu2017, per_layer_normalization=True)


def _dirty_fix(module, param_name, pruned_shape):
    module_param = getattr(module, param_name)

    # identify the dimension to prune
    pruned_dim = 0
    for original_size, pruned_size in zip(module_param.shape, pruned_shape):
        if original_size != pruned_size:
            keep_indices = torch.LongTensor(range(pruned_size)).to(module_param.data.device)
            module_param.data = module_param.data.index_select(pruned_dim, keep_indices)

            # modify number of features/channels
            if param_name == "weight":
                if isinstance(module, nn.modules.batchnorm._BatchNorm) or \
                        isinstance(module, nn.modules.instancenorm._InstanceNorm) or \
                        isinstance(module, nn.GroupNorm):
                    module.num_features = pruned_size
                elif isinstance(module, nn.modules.conv._ConvNd):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_channels = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_channels = pruned_size
                elif isinstance(module, nn.Linear):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_features = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_features = pruned_size
                else:
                    pass
        pruned_dim += 1


def load_pruned_model(model, pruned_weights, prefix='', load_pruned_weights=True, inplace=True):
    """load pruned weights to a unpruned model instance
    Arguments:
        model (pytorch model): the model instance
        pruned_weights (OrderedDict): pruned weights
        prefix (string optional): prefix (if has) of pruned weights
        load_pruned_weights (bool optional): load pruned weights to model according to the ICLR 2019 paper:
            "Rethinking the Value of Network Pruning", without finetuning, the model may achieve comparable or even
            better results
        inplace (bool, optional): if return a copy of the model
    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)
    """
    model_weight_names = model.state_dict().keys()
    pruned_weight_names = pruned_weights.keys()

    # check if module names match
    assert set([prefix + _ for _ in model_weight_names]) == set(pruned_weight_names)

    # inplace or return a new copy
    if not inplace:
        pruned_model = copy.deepcopy(model)
    else:
        pruned_model = model

    # update modules with mis-matched weight
    model_weights = model.state_dict()
    for model_weight_name in model_weight_names:
        if model_weights[model_weight_name].shape != pruned_weights[prefix + model_weight_name].shape:
            *container_names, module_name, param_name = model_weight_name.split('.')
            container = model
            for container_name in container_names:
                container = container._modules[container_name]
            module = container._modules[module_name]
            _dirty_fix(module, param_name, pruned_weights[prefix + model_weight_name].shape)
    if load_pruned_weights:
        pruned_model.load_state_dict({k: v for k, v in pruned_weights.items()})
    return pruned_model


def prune(model, input_shape, prune_ratio, prune_method=liu2017):
    """prune a model
    Arguments:
        model (pytorch model): the model instance
        input_shape (tuple): shape of the input tensor
        prune_ratio (float): ratio of be pruned channels to total channels
        prune_method (method): algorithm to prune weights
    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)
    Pipeline:
        1. generate mapping from tensors connected to BNs by parsing torch script traced graph
        2. identify corresponding BN and conv/linear like:
            conv/linear --> ... --> BN --> ... --> conv/linear
                                     |
                                    ...
                                     | --> relu --> ... --> conv/linear
                                    ...
                                     | --> ... --> maxpool --> ... --> conv/linear
            , where ... represents per channel operations. all the floating nodes must be conv/linear
        3. prune the weights of BN and connected conv/linear
        4. load weights to a unpruned model with pruned weights
    """
    # convert to CPU for simplicity
    src_device = next(model.parameters()).device
    model = model.cpu()

    # parse & generate mappings to BN layers
    prec_layers, succ_layers = get_pruning_layers(model, input_shape)

    # prune weights
    pruned_weights = prune_method(model.state_dict(), prune_ratio, prec_layers, succ_layers)

    # prune model according to pruned weights
    pruned_model = load_pruned_model(model, pruned_weights)

    return pruned_model.to(src_device)
```

netslim/sparse.py

```python
import torch
import torch.nn as nn


def update_bn(model, s=1e-4):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) or \
            isinstance(m, nn.modules.instancenorm._InstanceNorm) or \
            isinstance(m, nn.GroupNorm):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))


def update_bn_by_names(model, norm_layer_names, s=1e-4):
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        m.weight.grad.data.add_(s * torch.sign(m.weight.data))
```

