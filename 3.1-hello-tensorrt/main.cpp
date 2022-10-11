/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-09-29 09:20:36
 * @LastEditors: zwy
 * @LastEditTime: 2022-09-29 15:58:27
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// TensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda.h>
#include <cuda_runtime.h>

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity serverity, const char *msg) noexcept override
    {
        if (serverity <= Severity::kVERBOSE)
            std::cout << "[TRT] " << std::string(msg) << std::endl;
    }
};

nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights W;
    W.count = n;
    W.type = nvinfer1::DataType::kFLOAT;
    W.values = ptr;
    return W;
}
bool build_model(TRTLogger &logger, const char *model_file_path)
{
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    // 这是基本需要的组件
    //形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);
    // 构建一个模型
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */
    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    input->setName("input");
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    layer1->setName("FullyLayer1");
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    prob->setName("Sigmoid");
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    config->setMaxWorkspaceSize(1 << 28);
    builder->setMaxBatchSize(1); // 推理时 batchSize = 1

    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    // TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }
    // ----------------------------- 4. 序列化模型文件并存储 -----------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen(model_file_path, "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    //卸载顺序按转构建顺序的倒序
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    std::cout << "Done!" << std::endl;
    return true;
}

std::vector<uint8_t> load_file(const char *file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return {};
    }
    in.seekg(0, std::ios::end);
    size_t len = in.tellg();

    std::vector<uint8_t> data;
    if (len > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(len);
        in.read((char *)&data[0], len);
    }
    in.close();
    return data;
}

bool inference(TRTLogger &logger, const char *model_file_path)
{
    // ------------------------------ 1. 准备模型并加载   ----------------------------
    std::vector<uint8_t> engine_data = load_file(model_file_path);
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        std::cout << "Deserialize cuda engine failed." << std::endl;
        runtime->destroy();
        return false;
    }
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStreamCreate(&stream);
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */
    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {1, 2, 3};
    float *input_data_devive = nullptr;
    float output_data_host[2];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_devive, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));

    cudaMemcpyAsync(input_data_devive, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);
    // 用一个指针数组指定input和output在gpu中的指针。
    float *bindings[] = {input_data_devive, output_data_device};

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::cout << "Output data host -> [0]:" << output_data_host[0] << "  [1]:" << output_data_host[1] << std::endl;

    // ------------------------------ 4. 释放内存 ----------------------------
    std::cout << "Clean memory" << std::endl;
    cudaStreamDestroy(stream);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
    // ------------------------------ 5. 手动推理进行验证 ----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};
    std::cout << "手动验证计算结果：" << std::endl;
    for (int io = 0; io < num_output; io++)
    {
        float output_host = layer1_bias_values[io];
        for (int ii = 0; ii < num_input; ii++)
        {
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }
        // sigmoid
        float prob = 1 / (1 + exp(-output_host));
        std::cout << "output_prob[" << io << "] = " << prob << std::endl;
    }
    return true;
}

int main(int argc, char const *argv[])
{
    TRTLogger logger; // logger是必要的，用来捕捉warning和info等
    if (!build_model(logger, "engine.trtmodel"))
    {
        return -1;
    }
    if (!inference(logger, "engine.trtmodel"))
    {
        return -1;
    }
    return 0;
}
