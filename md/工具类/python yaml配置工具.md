# Python Yaml配置工具

【GiantPandaCV导语】深度学习调参过程中会遇到很多参数，为了完整保存一个项目的所有配置，推荐使用yaml工具进行配置。

## 简介

Yaml是可读的数据序列化语言，常用于配置文件。

支持类型有：

- 标量（字符串、证书、浮点）
- 列表 
- 关联数组 字典

语法特点：

- 大小写敏感
- 缩进表示层级关系
- 列表通过 "-" 表示，字典通过 ":"表示
- 注释使用 "#"

安装用命令：

```
pip install pyyaml
```

## 使用

举个例子：

```yaml
name: tosan
age: 22
skill:
  name1: coding
  time: 2years
job:
  - name2: JD
    pay: 2k
  - name3: HW
    pay: 4k
```

注意：关键字不能重复；不能使用tab，必须使用空格。

处理的脚本：

```python
import yaml 

f = open("configs/test.yml", "r")

y = yaml.load(f)

print(y)
```

输出结果：

```
YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  y = yaml.load(f)
{'name': 'tosan', 'age': 22, 'skill': {'name1': 'coding', 'time': '2years'}, 'job': [{'name2': 'JD', 'pay': '2k'}, {'name3': 'HW', 'pay': '4k'}]}
```

这个警告取消方法是：添加默认loader

```python
import yaml 

f = open("configs/test.yml", "r")

y = yaml.load(f, Loader=yaml.FullLoader)

print(y)
```

保存：

```python
content_dict = {
	'name':"ch",
}

f = open("./config.yml","w")

print(yaml.dump(content_dict, f))
```





## 语法

支持的类型：

```yaml
# 支持数字，整形、float
pi: 3.14 

# 支持布尔变量
islist: true
isdict: false

# 支持None 
cash: ~

# 时间日期采用ISO8601
time1: 2021-6-9 21:59:43.10-05:00

#强制转化类型
int_to_str: !!str 123
bool_to_str: !!str true

# 支持list
- 1
- 2
- 3

# 复合list和dict
test2:
  - name: xxx
    attr1: sunny
    attr2: rainy
    attr3: cloudy
```

