/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-09-29 15:29:31
 * @LastEditors: zwy
 * @LastEditTime: 2022-09-29 19:05:18
 */
#include <iostream>
#include <fstream>
#include <vector>

// tensorrt include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda.h>
#include <cuda_runtime.h>

// ----------------------------- 定义声明 -----------------------------
class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::cout << "[TRT]:" << std::string(msg) << std::endl;
        }
    }
};
nvinfer1::Weights make_weights(float *ptr, int size);
bool build_model(TRTLogger &logger, const char *file_save_path);
bool inference(TRTLogger &logger, const char *model_save_path);
std::vector<uint8_t> load_file(const char *);
// -------------------------------------------------------------------

/* 主函数 */
int main(int argc, char const *argv[])
{
    TRTLogger logger;
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

nvinfer1::Weights make_weights(float *ptr, int size)
{
    nvinfer1::Weights W;
    W.count = size;
    W.type = nvinfer1::DataType::kFLOAT;
    W.values = ptr;
    return W;
}
bool build_model(TRTLogger &logger, const char *file_save_path)
{
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);
    // 构建一个模型
    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */
    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1,
        0.1, 0.1, 0.1,
        0.2, 0.2, 0.2}; // 行优先
    float layer1_bias_values[] = {0.0};

    // 如果要使用动态shape，必须让NetworkDefinition的维度定义为-1，in_channel是固定的
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    input->setName("Input Layer");
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 1);

    nvinfer1::IConvolutionLayer *layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    layer1->setPadding(nvinfer1::DimsHW(1, 1));
    layer1->setName("Conv Leyer1");

    nvinfer1::IActivationLayer *prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));
    int maxBatchSize = 10;
    std::cout << "Workspace Size = " << (1 << 28) / 1024.0f / 1024.0f << std::endl;
    // 配置暂存存储器，用于layer实现的临时存储，也用于保存中间激活值
    config->setMaxWorkspaceSize(1 << 28);
    // --------------------------------- 2. 关于profile ----------------------------------
    // 如果模型有多个输入，则必须多个profile
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    // 配置最小允许1 x 1 x 3 x 3
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));
    // 配置最小允许1 x 1 x 5 x 5
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    config->addOptimizationProfile(profile);
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        std::cout << "Build engine failed." << std::endl;
        return false;
    }
    // -------------------------- 3. 序列化 ----------------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen(file_save_path, "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    std::cout << "Building model is Done!" << std::endl;
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
bool inference(TRTLogger &logger, const char *model_save_path)
{
    // ------------------------------- 1. 加载model并反序列化 -------------------------------
    std::vector<uint8_t> engine_data = load_file(model_save_path);
    if (engine_data.size() == 0)
    {
        std::cout << "loading engine file is failed!" << std::endl;
        return false;
    }
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        std::cout << "Deserialize cuda engine failed!" << std::endl;
        runtime->destroy();
        return false;
    }
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    /*
    Network definition:

    image
      |
    conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
      |
    relu
      |
    prob
    */
    // ------------------------------- 2. 输入与输出 -------------------------------
    float input_data_host[] = {
        // batch 0
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        // batch 1
        -1, 1, 1,
        1, 0, 1,
        1, 1, -1};
    float *input_data_device = nullptr;
    int ib = 2, iw = 3, ih = 3;
    float output_data_host[ib * iw * ih];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // ------------------------------- 3. 推理 -------------------------------
    // 明确当前推理时，使用的数据输入大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ------------------------------- 4. 输出结果 -------------------------------
    for (int i = 0; i < ib; i++)
    {
        std::cout << "batch:" << i << " => \n output data host : " << std::endl;
        for (int j = 0; j < iw * ih; j++)
        {
            std::cout << output_data_host[i * iw * ih + j] << " ";
            if ((j + 1) % iw == 0)
            {
                std::cout << std::endl;
            }
        }
    }
    std::cout << "Clean memory!" << std::endl;
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
}
