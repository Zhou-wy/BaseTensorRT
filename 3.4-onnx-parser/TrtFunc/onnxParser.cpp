/*
 * @Descripttion:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-02 10:38:11
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-03 10:45:04
 */

#include "onnxParser.h"

bool build_model(const char *onnx_path)
{
    TRTLogger logger;
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnx_path, 1))
    {
        std::cout << "Failed to parser demo.onnx" << std::endl;
        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // --------------------------------- 2.1 关于profile ----------------------------------
    // 如果模型有多个输入，则必须多个profile
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
    nvinfer1::ITensor *input_tensor = network->getInput(0);
    input_tensor->setName("Input");
    int input_channel = input_tensor->getDimensions().d[1];

    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));

    // 添加到配置
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    // -------------------------- 3. 序列化 ----------------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("building engine is done !");
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

bool inference(const char *model_save_path)
{
    TRTLogger logger;
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
        1, 9, 1,
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