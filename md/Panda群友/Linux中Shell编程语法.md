# Linux的Shell编程语法集锦



【GiantPandaCV导语】相信在linux服务器环境下完成算法开发和部署的同学，都有使用shell来实现部分自动化功能的经历，本文就来给大家分享我总结的一些shell语法知识，希望对大家有帮助。



## 一、shell文件运行

写好的文件保存为*.sh 文件加好运行权限后，就是可以用Bash运行的脚本程序了

```bash
chmod +x yourshell.sh
./yourshell.sh
```



## 二、shell 变量

命名格式 A=B 或用循环等语句给变量赋值

**注意：变量名和等号之间不能有空格**

```bash
使用时：echo $A或 echo ${A},花括号用来确认变量范围（可选）
只读变量： A=B readonly A
删除变量：unset A
```

变量类型：

1）局部变量  仅当前shell实例中有效

2）环境变量  全局的变量，比如用export声明的，或者在bashrc文件里或者/etc/profile文件里的

3）shell变量  由shell程序设置的特殊变量



**shell 字符串**

A='B' 或 A="B"

注意：单引号中的变量是无效的，双引号中的可使用转义字符

> 字符串拼接
A=B
C="D,""$A"
E="F,${A}"

> 获取字符串长度
A=B
echo ${#A}

> 提取子串
A=B
echo ${A:1:4}

> 查找子串
A=B  此处的是反引号不是单引号
echo `expr index "$A"`



**shell 数组**

bash仅支持一维数组
A=(B C D E F)  或A[0]=B A[1]=E A[3]=F 可以不使用连续下标
读取  ${A[i]}  ${A[@]} 表示获取所有元素



shell 传递参数

- $0 脚本名
- $1-$9 输入脚本的参数，第一个、第二个以此类推
- $@ 所有的参数
- $# 参数数量
- $？返回上一条指令的代码
- $$ 当前脚本的进程标识号（PID）
- !! 完整的last命令，包括参数。
- $_最后一条指令的最后一个参数


## 三、shell 基础运算符

**算数运算符**
shell原生不支持数学计算，可通过awk或expr实现，或者使用(())，在内层小括号内，使用C的语法实现。使用expr时，`为反引号而不是单引号

```bash
val=`expr 2 + 2`
```

注意：表达书和运算符之间必须有空格，条件表达式要放在方括号内，例如[$a == $b]，乘号前必须加反斜杠

**关系运算符**

```bash

注意：只支持数字，不支持字符串，除非字符串的值是数字
-eq  是否相等
-ne  是否不相等
-gt  左边是否大于右边
-lt   左边是否小于右边
-ge 左边是否大于等于右边 
-le  左边是否小于等于右边
```

**布尔运算符**

```bash
! 非
-o 或
-a 与
```

**逻辑运算符**

```bash
&&　逻辑的AND
||　逻辑的OR
```



**字符运算符**

```bash
= 相等
!= 不相等
-z 长度为0
-n 不为0长度
$  是否为空
```

**文件测试运算符**

```bash
文件测试运算符用于检测 Unix 文件的各种属性。
-b file	检测文件是否是块设备文件，如果是，则返回 true。	  [ -b $file ] 返回 false。
-c file	检测文件是否是字符设备文件，如果是，则返回 true。	[ -c $file ] 返回 false。
-d file	检测文件是否是目录，如果是，则返回 true。	        [ -d $file ] 返回 false。
-f file	检测文件是否是普通文件（既不是目录，也不是设备文件），如果是，则返回 true。	[ -f $file ] 返回 true。
-g file	检测文件是否设置了 SGID 位，如果是，则返回 true。	[ -g $file ] 返回 false。
-k file	检测文件是否设置了粘着位(Sticky Bit)，如果是，则返回 true。	[ -k $file ] 返回 false。
-p file	检测文件是否是有名管道，如果是，则返回 true。	    [ -p $file ] 返回 false。
-u file	检测文件是否设置了 SUID 位，如果是，则返回 true。	[ -u $file ] 返回 false。
-r file	检测文件是否可读，如果是，则返回 true。	          [ -r $file ] 返回 true。
-w file	检测文件是否可写，如果是，则返回 true。	          [ -w $file ] 返回 true。
-x file	检测文件是否可执行，如果是，则返回 true。	        [ -x $file ] 返回 true。
-s file	检测文件是否为空（文件大小是否大于0），不为空返回 true。	[ -s $file ] 返回 true。
-e file	检测文件（包括目录）是否存在，如果是，则返回 true。	[ -e $file ] 返回 true。
```

## 四、shell echo

```bash
read name 类似python的input函数，可以用来获取输入值
echo -e "ok! \n" 
-e 开启转义
\c 不换行
\n 换行
```

## 五、shell printf

```bash
语法
printf format-string [arguments...]

举例：
printf "test\n"
```

## 六、shell test

```bash
用于检查某个条件是否成立，可进行数值、字符和文件三方面的测试

数值测试
-eq	等于则为真
-ne	不等于则为真
-gt	大于则为真
-ge	大于等于则为真
-lt	小于则为真
-le	小于等于则为真
num1=100
num2=100
if test $[num1] -eq $[num2]
then
    echo '两个数相等！'
else
    echo '两个数不相等！'
fi

代码内的[]中可进行基本的数值运算

字符串测试
=	等于则为真
!=	不相等则为真
-z 字符串	字符串的长度为零则为真
-n 字符串	字符串的长度不为零则为真
num1="alasijia"
num2="alasi1jia"

if test $num1 = $num2
then
    echo '两个字符串相等!'
else
    echo '两个字符串不相等!'
fi

文件测试
-e 文件名	如果文件存在则为真
-r 文件名	如果文件存在且可读则为真
-w 文件名	如果文件存在且可写则为真
-x 文件名	如果文件存在且可执行则为真
-s 文件名	如果文件存在且至少有一个字符则为真
-d 文件名	如果文件存在且为目录则为真
-f 文件名	如果文件存在且为普通文件则为真
-c 文件名	如果文件存在且为字符型特殊文件则为真
-b 文件名	如果文件存在且为块特殊文件则为真
cd /bin
if test -e ./bash
then
    echo '文件已存在!'
else
    echo '文件不存在!'
fi
```

## 七、shell 流程控制

```bash
if condition
then
   command
else
fi

if condition
then 
   command
elif condition2
then
   command
else
   command
fi

for var in item1 item2
do
   command
done

while condition
do 
  command 
done
```

## 八、函数

```bash
[function] funname[()]
{
  action;
  [return int;]
}

fun(){
action
｝
```

## 九、当前脚本包含其他脚本

```bash
source tesh.sh 或 . tesh.sh
```

## 十、shell中双括号，双中括号的含义

详见： https://www.jb51.net/article/123081.htm




## 十一、示例demo

这里引用MIT课程里的一个脚本，该课程名称为《The Missing Semester of Your CS Education》大家可以上网搜索的到，加上注释帮助大家理解

```bash
#!/bin/bash 
echo $(ls)    # 执行ls命令并打印执行结果，这里就是打印当前文件夹下的所有文件
echo "Starting program at $(date)" # 这句会首先执行data指令来获取当前时间信息并将该信息重定向到当前语句，再打印输出
echo "Running program $0 with $# arguments with pid $$"  # 此处的$0会重定向为脚本名，$#会重定向为当前参数数量 $$为当前执行脚本的进程标识号
# 此处就是一个for循环，一个一个的拿出调用脚本时传入的参数
for file in "$@"; do    
    grep foobar "$file" > /dev/null 2> /dev/null    # 这里的grep函数是有返回值的，这里的意思是，返回有或者无时重定向到指定位置，无该文件时重定向到另一指定位置
    # 这里要注意，中括号和里面的执行指令之间一定要有空格
    if [[ $? -ne 0 ]]; then        
       echo "File $file does not have any foobar, adding one"        
       echo "# foobar" >> "$file"    
       fi 
done
```



