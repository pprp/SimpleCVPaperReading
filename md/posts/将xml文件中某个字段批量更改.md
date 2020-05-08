---
title: 将xml文件中某个字段批量更改
date: 2019-10-15 14:54:46
tags: 
- python
- xml
- 批处理
---





# 将xml文件中某个字段批量更改

> 前言：使用labelimg进行标注的时候，由于都是用的是默认的名称，有时候类的名字会出现拼写错误，比如我想要写的是“cow” 结果打上去的是“cwo”, 一出错就错一片，这很常见，所以参考了：<https://www.jianshu.com/p/cf12bef0872c> 的代码，修改了冗余的代码，并添加了新的模块以后，将代码分享给大家。



## 1. xml文件展示

```xml
<annotation>
	<folder>part3</folder>
	<filename>0015-1150.jpg</filename>
	<path>I:\part3\0015-1150.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1280</width>
		<height>720</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>459</xmin>
			<ymin>88</ymin>
			<xmax>567</xmax>
			<ymax>163</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>543</xmin>
			<ymin>92</ymin>
			<xmax>630</xmax>
			<ymax>163</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>659</xmin>
			<ymin>156</ymin>
			<xmax>773</xmax>
			<ymax>263</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>497</xmin>
			<ymin>171</ymin>
			<xmax>677</xmax>
			<ymax>315</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>584</xmin>
			<ymin>303</ymin>
			<xmax>715</xmax>
			<ymax>395</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>537</xmin>
			<ymin>389</ymin>
			<xmax>814</xmax>
			<ymax>497</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>750</xmin>
			<ymin>303</ymin>
			<xmax>918</xmax>
			<ymax>434</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>770</xmin>
			<ymin>223</ymin>
			<xmax>958</xmax>
			<ymax>285</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>974</xmin>
			<ymin>477</ymin>
			<xmax>1152</xmax>
			<ymax>636</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>66</xmin>
			<ymin>454</ymin>
			<xmax>308</xmax>
			<ymax>696</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>347</xmin>
			<ymin>548</ymin>
			<xmax>550</xmax>
			<ymax>720</ymax>
		</bndbox>
	</object>
	<object>
		<name>cow</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>996</xmin>
			<ymin>154</ymin>
			<xmax>1128</xmax>
			<ymax>245</ymax>
		</bndbox>
	</object>
</annotation>
```

## 2. 将文件夹中所有name进行批量更改

### 2.1 将某类别名称更改

```python
def changeName(xml_fold, origin_name, new_name):
    '''
    xml_fold: xml存放文件夹
    origin_name: 原始名字，比如弄错的名字，原先要cow,不小心打成cwo
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    files = os.listdir(xml_fold)
    cnt = 0 
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):#获取object节点中的name子节点
            tmp_name = obj.find('name').text
            if tmp_name == origin_name: # 修改
                obj.find('name').text = new_name
                print("change %s to %s." % (origin_name, new_name))
                cnt += 1
        dom.write(file_path, xml_declaration=True)#保存到指定文件
    print("有%d个文件被成功修改。" % cnt)
```

### 2.2 将某文件夹所有类别都统一成某个类别

适用于只有一个类，并且进行改名的时候。

```python
def changeAll(xml_fold,new_name):
    '''
    xml_fold: xml存放文件夹
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    files = os.listdir(xml_fold)
    cnt = 0 
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):#获取object节点中的name子节点
            tmp_name = obj.find('name').text
            obj.find('name').text = new_name
            print("change %s to %s." % (tmp_name, new_name))
            cnt += 1
        dom.write(file_path, xml_declaration=True)#保存到指定文件
    print("有%d个文件被成功修改。" % cnt)
```

### 2.3 统计每个类别实际目标的个数

```python
def countAll(xml_fold):
    '''
    xml_fold: xml存放文件夹
    '''
    files = os.listdir(xml_fold)
    dict={}
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):#获取object节点中的name子节点
            tmp_name = obj.find('name').text
            if tmp_name not in dict:
                dict[tmp_name] = 0
            else:
                dict[tmp_name] += 1
        dom.write(file_path, xml_declaration=True)#保存到指定文件
    print("统计结果如下：")
    print("-"*10)
    for key,value in dict.items():
        print("类别为%s的目标个数为%d." % (key, value))
    print("-"*10)
```

## 3. 全部代码

```python
import os
import os.path
from xml.etree.ElementTree import parse, Element

def changeName(xml_fold, origin_name, new_name):
    '''
    xml_fold: xml存放文件夹
    origin_name: 原始名字，比如弄错的名字，原先要cow,不小心打成cwo
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    files = os.listdir(xml_fold)
    cnt = 0 
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):#获取object节点中的name子节点
            tmp_name = obj.find('name').text
            if tmp_name == origin_name: # 修改
                obj.find('name').text = new_name
                print("change %s to %s." % (origin_name, new_name))
                cnt += 1
        dom.write(file_path, xml_declaration=True)#保存到指定文件
    print("有%d个文件被成功修改。" % cnt)

def changeAll(xml_fold,new_name):
    '''
    xml_fold: xml存放文件夹
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    files = os.listdir(xml_fold)
    cnt = 0 
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):#获取object节点中的name子节点
            tmp_name = obj.find('name').text
            obj.find('name').text = new_name
            print("change %s to %s." % (tmp_name, new_name))
            cnt += 1
        dom.write(file_path, xml_declaration=True)#保存到指定文件
    print("有%d个文件被成功修改。" % cnt)

def countAll(xml_fold):
    '''
    xml_fold: xml存放文件夹
    '''
    files = os.listdir(xml_fold)
    dict={}
    for xmlFile in files:
        file_path = os.path.join(xml_fold, xmlFile)
        dom = parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):#获取object节点中的name子节点
            tmp_name = obj.find('name').text
            if tmp_name not in dict:
                dict[tmp_name] = 0
            else:
                dict[tmp_name] += 1
        dom.write(file_path, xml_declaration=True)#保存到指定文件
    print("统计结果如下：")
    print("-"*10)
    for key,value in dict.items():
        print("类别为%s的目标个数为%d." % (key, value))
    print("-"*10)


if __name__ == '__main__':
    path = r"I:\dongpeijiePickup\assignment\part2_xml" #xml文件所在的目录
    # changeName(path, "cattle", "cow")
    # changeAll(path, "cattle")
    countAll(path)

```

