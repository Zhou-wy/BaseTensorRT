<!--
 * @Descripttion: 
 * @version: 
 * @Author: zwy
 * @Date: 2022-09-29 14:59:20
 * @LastEditors: zwy
 * @LastEditTime: 2022-09-29 15:02:07
-->
## 构建模型，生成engine
- main.cpp 构建了一个最简单全连接网络
- tensorrt的工作流程如下图：
  - 首先定义网络
  - 优化builder参数
  - 通过builder生成engine,用于模型保存、推理等
  - engine可以通过序列化和逆序列化转化模型数据类型（转化为二进制byte文件，加快传输速率），再进一步推动模型由输入张量到输出张量的推理。
  ![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/1.simple-fully-connected-network.jpg)
   ![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/2.tensortr-workflow.jpg)
- code structure
   1. 定义 builder, config 和network，其中builder表示所创建的构建器，config表示创建的构建配置（指定TensorRT应该如何优化模型），network为创建的网络定义。
   2. 输入，模型结构和输出的基本信息（如下图所示）
    ![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/2.tensortr-workflow.jpg)
   1. 生成engine模型文件
   2. 序列化模型文件并存储

## 读取engine，完成inference
执行推理的步骤：
  1. 准备模型并加载
  2. 创建runtime：`createInferRuntime(logger)`
  3. 使用运行时时，以下步骤：
     1. 反序列化创建engine, 得为engine提供数据：`runtime->deserializeCudaEngine(modelData, modelSize)`,其中`modelData`包含的是input和output的名字，形状，大小和数据类型
        ```cpp
        class ModelData(object):
        INPUT_NAME = "data"
        INPUT_SHAPE = (1, 1, 28, 28) // [B, C, H, W]
        OUTPUT_NAME = "prob"
        OUTPUT_SIZE = 10
        DTYPE = trt.float32
        ```

     2. 从engine创建执行上下文:`engine->createExecutionContext()`
  4. 创建CUDA流`cudaStreamCreate(&stream)`：
     1. CUDA编程流是组织异步工作的一种方式，创建流来确定batch推理的独立
     2. 为每个独立batch使用IExecutionContext(3.2中已经创建了)，并为每个独立批次使用cudaStreamCreate创建CUDA流。
     
  5. 数据准备：
     1. 在host上声明`input`数据和`output`数组大小，搬运到gpu上
     2. 要执行inference，必须用一个指针数组指定`input`和`output`在gpu中的指针。
     3. 推理并将`output`搬运回CPU
  6. 启动所有工作后，与所有流同步以等待结果:`cudaStreamSynchronize`
  7. 按照与创建相反的顺序释放内存