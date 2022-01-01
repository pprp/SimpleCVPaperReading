# Atlas基础知识

> 本文介绍在华为Atlas上部署自己的算法时需要配置的基础环境以及基本的api函数使用讲解

# 1、硬件介绍及安装操作系统

我使用的是Atlas  500 pro，官网提供的彩页链接如下：

[https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-500-pro-3000](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/atlas-500-pro-3000)

华为Atlas上支持的操作系统大概是三种：华为自有的操作系统、centos7.6和ubuntu18.5（要看自己用的机器的文档来更新）。然后安装操作系统的教程为：

[准备硬件环境_CANN商用版(3.3.0) 环境部署_软件安装指南 (通过msInstaller工具) _安装前准备_A500 Pro-3000_华为云](https://support.huaweicloud.com/instg-msInstaller-cann330/atlasmsin_03_0034.html)

# 2、软件安装

华为的加速卡有多种型号，然后提供的驱动和固件版本也在持续更新。所以有时候要保持软件的更新。华为提供的软件包主要包括5个：

- driver&firmware 华为推理加速卡的驱动和固件，安装顺序为先driver后firmware，且这两个包必须最先安装。卸载顺序为driver后firmware
- toolkit 华为的离线推理引擎包，推理引擎的头文件和库都是用这个包安装的，华为的模型转换工具也在这个包里
- nnrt     也是华为提供的离线推理引擎包，和toolkit的区别是，这个包可以在容器内安装，然后在容器内用于调用推理引擎使用
- toolbox华为提供的实用工具包，一些查看系统版本，卡状态等工具性插件都在这个包里

安装方式分为华为工具安装和自己手动安装两种方式，可参考文档配置：

[安装须知_ CANN社区版安装指南(5.0.2.alpha003)_软件安装指南 (开发&运行场景, 通过命 令行方式)_华为云](https://support.huaweicloud.com/instg-cli-cann502-alpha003/atlasdeploy_03_0002.html)

# 3、模型转换工具ATC

[ATC工具使用环境搭建_昇腾CANN社区版(5.0.2.alpha003)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0004.html)

# 4、C++推理引擎代码编译知识点

> 推荐使用cmake进行编译配置，因为我只会cmake，hhh

首先放华为的文档链接：

[编译及运行应用_昇腾CANN社区版(5.0.2.alpha003)(推理)_应用开发（C++）_华为云](https://support.huaweicloud.com/aclcppdevg-cann502alpha3infer/atlasdevelopment_01_0109.html)

然后说一下自己的理解：

华为Atlas的机器学习模块ACL其实相对来说，编译难度很低了，没有很多复杂的配置要求，需要什么编译什么所以这里也没啥说的啦。

# 5、C++基础函数API讲解和使用逻辑举例

**头文件** 

```cpp
#include "acl/acl/h" // 华为的基础组件头文件，必须要包括的，剩余的，可以参考上面的文档自由选择
```

**常用API函数及结构体讲解讲解**

```cpp
// **结构体** aclError 华为定义的错误类型的宏，基本就是0代表正常，其余值代表各种各样的问题及保留值
aclError ret = aclInit(); // AscendCL 的初始化代码，必须在调用所有推理部分前调用
ret = aclFinalize(); // AscendCL去初始化函数，这个函数执行了以后整个ACL模块就停掉了

// 模型加载流程
// **创建流程**
// 基本创建流程组件包括
// Device  指定当前模型的推理卡ID
ret = aclrtSetDevice(deviceid); //指定后续加载模型时加载到哪张推理卡上
// context 模型的上下文管理容器
aclrtContext context;
ret = aclrtCreatecontext(&context,deviceid);
// stream  用来维护一些异步操作的执行顺序
aclrtstream stream;
ret = aclrtCreateStream(&stream);
// runmode 当前指定模型的运行模式，分为HOST端模式和DEVICE端模式两种
aclrtRunMode runmode;
ret = aclrtGetRunMode(&runmode);
// modelID 通过device id和model id来代表某个具体的加载模型
// 模型加载有多种方式，比如从内存加载、从文件加载、自己管理模型内存或者交由acL模块管理模型内存等
// 这里提供的是从文件加载模型，并交由acl管理内存的方式
int modelIdl;
model_file = "***.om";
ret = aclmdlLoadFromFile(model_file.c_str(),&modelId);
// modelDesc 模型描述结构体，其中包括模型输入端个数，输出端个数等信息
aclmdlDesc *modelDesc
model = aclmdlCreateDesc();
ret = aclmdlGetDesc(modelDesc,modelId);

// **析构流程**
// 析构顺序为 用modelid卸载模型、析构modelDesc、重置推理卡及关闭acl服务
ret = aclmdlUnload(modelId);
ret = aclmdlDestroyDesc(modelDesc);
ret = aclrtResetDevice(deviceId_);
ret = aclFinalize(); // 理论上后两步不应该在推理引擎内，而应该在调度器上完成

// 模型推理流程
// **创建流程**
// 模型的推理输入需要每次进行创建，推理输出的buffer可以一次性创建并一直复用
//// 创建输入
size_t modelInputSize;
void *modelInputBuffer = nullptr;
modelInputSize = aclmdlGetInputSizeByIndex(modelDesc, 0);
// 申请一块HOST侧的内存        
ret = aclrtMalloc(&modelInputBuffer, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
       
// 创建aclmdlDataset类型的数据，描述模型推理的输入，input_为aclmdlDataset类型
// 利用之前创建的HOST端内存，在DEVICE侧挂载，然后把自己的数据赋值到该块内存就完成的模型输入准备
aclmdlDataset *input_;
input_ = aclmdlCreateDataset();
aclDataBuffer *inputData = aclCreateDataBuffer(modelInputBuffer, modelInputSize);
ret = aclmdlAddDatasetBuffer(input_, inputData);
ret = aclrtMemcpy(modelInputBuffer, modelInputSize, input_host_memory_+i*3*yolo_params_.INPUT_H*yolo_params_.INPUT_W,yolo_params_.INPUT_H*yolo_params_.INPUT_W*3*sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
//// 创建输出
aclmdlDataset *output_;
size_t outputSize = aclmdlGetNumOutputs(modelDesc);
output_ = aclmdlCreateDataset();
// 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中
// 动态batch下，output为按最大batch值创建的内存块
for (size_t i = 0; i < outputSize; ++i) {
      size_t buffer_size = aclmdlGetOutputSizeByIndex(modelDesc, i);
      void *outputBuffer = nullptr;
      ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
      if(ret != 0)
      {
          return -2;
      }
      aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);   
      ret = aclmdlAddDatasetBuffer(output_, outputData);
      if(ret != 0)
      {
          return -2;
      }
}
//// 执行推理
ret = aclmdlExecute(modelId, input_, output_);
//// 获取输出到host侧
aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, idx);
    
// 获取buffer地址
void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    
// 获取buffer的长度
size_t bufferSize = aclGetDataBufferSizeV2(dataBuffer);
    
// 将指定内存从device拷贝到host的内存上，此时buffer内存即为模型多个推理输出中的指定索引输出
void* buffer = new uint8_t[bufferSize]; 
aclError aclRet = aclrtMemcpy(buffer, bufferSize, dataBufferDev, bufferSize, ACL_MEMCPY_DEVICE_TO_HOST);
// **析构流程**
这里的析构是说，在每次执行完之后，都要释放掉input重新创建，因为input部分不可复用
if (input_ != nullptr) 
        {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) 
            {
              aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
              aclDestroyDataBuffer(dataBuffer);
            }
            aclmdlDestroyDataset(input_);
            input_ = nullptr;
        }

// 输出部分的析构可以放在类的析构时再做
if (output_ != nullptr) {
        // 此处应该写入日志
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) 
        {
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(dataBuffer);
           (void)aclrtFree(data);
           (void)aclDestroyDataBuffer(dataBuffer);
        }
        (void)aclmdlDestroyDataset(output_);
        output_ = nullptr;
    }
```

**已知特殊技巧讲解**

- AIPP使用技巧及注意点

    > AIPP是华为提供的AI预处理，用于在AI Core上完成图像预处理，包括改变图像尺寸、色域转换（转换图像格式）、减均值/乘系数（改变图像像素），数据处理之后再进行真正的模型推理。

    [什么是AIPP_昇腾CANN社区版(5.0.2.alpha003)(推理)_ATC模型转换_高级功能_AIPP使能_华为云](https://support.huaweicloud.com/atctool-cann502alpha3infer/atlasatc_16_0015.html)

- 动态batch支持使用技巧及注意点

    > 动态batch推理就是通过在华为模型的输入端增加一个输入值来指定当前模型一次性推理图像数量来，从而调用batch推理能力

    1、首先在转换模型时，需要增加转换参数

     在使用华为提供的ATC模型转换工具时，需要修改参数—input_shape ="data:-1,3,416,416"把batch值设置为-1，然后增加一个参数 —dynamic_batch_size="1,2,4,8"，这里面的值就是设置的不同档位，也就是允许的一个batch送入图像数量，好像一次设置档位最多只能有32个值

     2、其次是在写推理代码时，需要增加一段对模型当前batch图像数的描述

    ```cpp
    size_t inputLen = aclmdlGetInputSizeByIndex(modelDesc, 1);
    void *data = nullptr;
    aclError ret = aclrtMalloc(&data, inputLen, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *dataBuffer = aclCreateDataBuffer(data, inputLen);
    ret = aclmdlAddDatasetBuffer(input_, dataBuffer);
    
    size_t index;
    //获取动态Batch输入的index，标识动态Batch输入的输入名称固定为ACL_DYNAMIC_TENSOR_NAME
    ret = aclmdlGetInputIndexByName(modelDesc, ACL_DYNAMIC_TENSOR_NAME, &index);
    // 设置当前batch的图像数量，必须是模型转换时设置的值
    ret = aclmdlSetDynamicBatchSize(modelId, input_, index, one_batch_size);
    ```

     3、推理输入数据准备时的注意点,通过指针偏移的形式来完成输入数据到模型指定input的buffer的拷贝

    ```cpp
    aclError ret = aclrtMemcpy(modelInputBuffer, 416*416*3, inputBuff, inputBuffSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    ret = aclrtMemcpy(modelInputBuffer + 416*416*3, 416*416*3, Img1.data, inputBuffSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    ```

     4、获取模型输出数据时的注意点

    以下部分为重点内容，请认真看完

    1、动态batch情况下，模型的输出空间会按照模型档位设置里的最大值创建，所以注意你的内存不要爆掉

    2、动态batch输出时，你获取的内存中是按照第一张图的值-》第二张图的值的顺序得到的

    3、注意模型的输出format，c++代码的输出是一维数组，是按照你的模型指定形状最后一维开始往前遍历，拉成一维数组输出的。

# 6、样例

最后提供一个华为官方的基于resnet50的分类算法样例：

https://support.huaweicloud.com/aclcppdevg-cann502alpha3infer/atlasdevelopment_01_0005.html

**剩余待探索高级技巧清单**

- [ ]  Context及stream管理，及切卡操作
- [ ]  动态分辨率支持
- [ ]  调度器编写