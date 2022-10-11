/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-09 13:35:56
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-09 20:00:12
 */
// CUDA include
#include <cuda_runtime.h>
#include <cublas_v2.h>

// TensorRt include
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

// system include
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <sstream>
#include <stdarg.h>

// Personal include
#include "utils/cuda_tools.cuh"
#include "utils/simple_logger.hpp"
#include "trt_builder.hpp"

class Logger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {

        if (severity == Severity::kINTERNAL_ERROR)
        {
            INFOE("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        }
        else if (severity == Severity::kERROR)
            INFOE("NVInfer: %s", msg);
        else if (severity == Severity::kWARNING)
            INFOW("NVInfer: %s", msg);
        else if (severity == Severity::kINFO)
            INFOD("NVInfer: %s", msg);
        else
            INFOD("%s", msg);
    };
};

static Logger gLogger;

namespace TRT
{
    /**
     * @description:将Tensor维度转成字符串类型输出
     * @param {vector<int>} &dims:Tensor 维度如[-1, 3, 112, 112]
     * @return {*}: -1x3x112x112
     */
    static std::string join_dims(const std::vector<int> &dims) //[-1,3,112,112]
    {
        std::stringstream output;
        char buf[64];
        const char *fmts[] = {"%d", " x %d"};
        for (int i = 0; i < dims.size(); ++i)
        {
            snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
            output << buf;
        }
        return output.str();
    }

    /**
     * @description: 保存文件
     * @param {string} &file : 保存文件路径
     * @param {void} *data: 写入文件的数据
     * @param {size_t} length: 写入数据的长度
     * @return {*}: 判断是否写入成功
     */
    bool save_file(const std::string &file, const void *data, size_t length)
    {

        FILE *f = fopen(file.c_str(), "wb");
        if (!f)
            return false;
        if (data && length > 0)
        {
            if (fwrite(data, 1, length, f) not_eq length)
            {
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }

    /**
     * @description:格式化字符串
     * @param {char} *fmt: 字符串序列
     * @return {*}:
     */
    static std::string format(const char *fmt, ...)
    {
        va_list vl;
        /*C 库宏 void va_start(va_list ap, last_arg) 初始化 ap 变量，
        它与 va_arg 和 va_end 宏是一起使用的。last_arg 是最后一个传递给函数的已知的固定参数，即省略号之前的参数。
        这个宏必须在使用 va_arg 和 va_end 之前被调用。*/
        va_start(vl, fmt);
        char buffer[10000];
        vsprintf(buffer, fmt, vl);
        return buffer;
    }

    static std::string dims_str(const nvinfer1::Dims &dims)
    {
        return join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
    }

    static const char *padding_mode_name(nvinfer1::PaddingMode mode)
    {
        switch (mode)
        {
        case nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN:
            return "explicit round down";
        case nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP:
            return "explicit round up";
        case nvinfer1::PaddingMode::kSAME_UPPER:
            return "same supper";
        case nvinfer1::PaddingMode::kSAME_LOWER:
            return "same lower";
        case nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN:
            return "caffe round down";
        case nvinfer1::PaddingMode::kCAFFE_ROUND_UP:
            return "caffe round up";
        }
        return "Unknow padding mode";
    }

    static const char *pooling_type_name(nvinfer1::PoolingType type)
    {
        switch (type)
        {
        case nvinfer1::PoolingType::kMAX:
            return "MaxPooling";
        case nvinfer1::PoolingType::kAVERAGE:
            return "AveragePooling";
        case nvinfer1::PoolingType::kMAX_AVERAGE_BLEND:
            return "MaxAverageBlendPooling";
        }
        return "Unknow pooling type";
    }

    static const char *activation_type_name(nvinfer1::ActivationType activation_type)
    {
        switch (activation_type)
        {
        case nvinfer1::ActivationType::kRELU:
            return "ReLU";
        case nvinfer1::ActivationType::kSIGMOID:
            return "Sigmoid";
        case nvinfer1::ActivationType::kTANH:
            return "TanH";
        case nvinfer1::ActivationType::kLEAKY_RELU:
            return "LeakyRelu";
        case nvinfer1::ActivationType::kELU:
            return "Elu";
        case nvinfer1::ActivationType::kSELU:
            return "Selu";
        case nvinfer1::ActivationType::kSOFTSIGN:
            return "Softsign";
        case nvinfer1::ActivationType::kSOFTPLUS:
            return "Parametric softplus";
        case nvinfer1::ActivationType::kCLIP:
            return "Clip";
        case nvinfer1::ActivationType::kHARD_SIGMOID:
            return "Hard sigmoid";
        case nvinfer1::ActivationType::kSCALED_TANH:
            return "Scaled tanh";
        case nvinfer1::ActivationType::kTHRESHOLDED_RELU:
            return "Thresholded ReLU";
        }
        return "Unknow activation type";
    }

    static std::string layer_type_name(nvinfer1::ILayer *layer)
    {
        switch (layer->getType())
        {
        case nvinfer1::LayerType::kCONVOLUTION:
            return "Convolution";
        case nvinfer1::LayerType::kFULLY_CONNECTED:
            return "Fully connected";
        case nvinfer1::LayerType::kACTIVATION:
        {
            nvinfer1::IActivationLayer *act = (nvinfer1::IActivationLayer *)layer;
            auto type = act->getActivationType();
            return activation_type_name(type);
        }
        case nvinfer1::LayerType::kPOOLING:
        {
            nvinfer1::IPoolingLayer *pool = (nvinfer1::IPoolingLayer *)layer;
            return pooling_type_name(pool->getPoolingType());
        }
        case nvinfer1::LayerType::kLRN:
            return "LRN";
        case nvinfer1::LayerType::kSCALE:
            return "Scale";
        case nvinfer1::LayerType::kSOFTMAX:
            return "SoftMax";
        case nvinfer1::LayerType::kDECONVOLUTION:
            return "Deconvolution";
        case nvinfer1::LayerType::kCONCATENATION:
            return "Concatenation";
        case nvinfer1::LayerType::kELEMENTWISE:
            return "Elementwise";
        case nvinfer1::LayerType::kPLUGIN:
            return "Plugin";
        case nvinfer1::LayerType::kUNARY:
            return "UnaryOp operation";
        case nvinfer1::LayerType::kPADDING:
            return "Padding";
        case nvinfer1::LayerType::kSHUFFLE:
            return "Shuffle";
        case nvinfer1::LayerType::kREDUCE:
            return "Reduce";
        case nvinfer1::LayerType::kTOPK:
            return "TopK";
        case nvinfer1::LayerType::kGATHER:
            return "Gather";
        case nvinfer1::LayerType::kMATRIX_MULTIPLY:
            return "Matrix multiply";
        case nvinfer1::LayerType::kRAGGED_SOFTMAX:
            return "Ragged softmax";
        case nvinfer1::LayerType::kCONSTANT:
            return "Constant";
        case nvinfer1::LayerType::kRNN_V2:
            return "RNNv2";
        case nvinfer1::LayerType::kIDENTITY:
            return "Identity";
        case nvinfer1::LayerType::kPLUGIN_V2:
            return "PluginV2";
        case nvinfer1::LayerType::kSLICE:
            return "Slice";
        case nvinfer1::LayerType::kSHAPE:
            return "Shape";
        case nvinfer1::LayerType::kPARAMETRIC_RELU:
            return "Parametric ReLU";
        case nvinfer1::LayerType::kRESIZE:
            return "Resize";
        }
        return "Unknow layer type";
    }

    static std::string layer_descript(nvinfer1::ILayer *layer)
    {
        switch (layer->getType())
        {
        case nvinfer1::LayerType::kCONVOLUTION:
        {
            nvinfer1::IConvolutionLayer *conv = (nvinfer1::IConvolutionLayer *)layer;
            return format("channel: %d, kernel: %s, padding: %s, stride: %s, dilation: %s, group: %d",
                          conv->getNbOutputMaps(),
                          dims_str(conv->getKernelSizeNd()).c_str(),
                          dims_str(conv->getPaddingNd()).c_str(),
                          dims_str(conv->getStrideNd()).c_str(),
                          dims_str(conv->getDilationNd()).c_str(),
                          conv->getNbGroups());
        }
        case nvinfer1::LayerType::kFULLY_CONNECTED:
        {
            nvinfer1::IFullyConnectedLayer *fully = (nvinfer1::IFullyConnectedLayer *)layer;
            return format("output channels: %d", fully->getNbOutputChannels());
        }
        case nvinfer1::LayerType::kPOOLING:
        {
            nvinfer1::IPoolingLayer *pool = (nvinfer1::IPoolingLayer *)layer;
            return format(
                "window: %s, padding: %s",
                dims_str(pool->getWindowSizeNd()).c_str(),
                dims_str(pool->getPaddingNd()).c_str());
        }
        case nvinfer1::LayerType::kDECONVOLUTION:
        {
            nvinfer1::IDeconvolutionLayer *conv = (nvinfer1::IDeconvolutionLayer *)layer;
            return format("channel: %d, kernel: %s, padding: %s, stride: %s, group: %d",
                          conv->getNbOutputMaps(),
                          dims_str(conv->getKernelSizeNd()).c_str(),
                          dims_str(conv->getPaddingNd()).c_str(),
                          dims_str(conv->getStrideNd()).c_str(),
                          conv->getNbGroups());
        }
        case nvinfer1::LayerType::kACTIVATION:
        case nvinfer1::LayerType::kPLUGIN:
        case nvinfer1::LayerType::kLRN:
        case nvinfer1::LayerType::kSCALE:
        case nvinfer1::LayerType::kSOFTMAX:
        case nvinfer1::LayerType::kCONCATENATION:
        case nvinfer1::LayerType::kELEMENTWISE:
        case nvinfer1::LayerType::kUNARY:
        case nvinfer1::LayerType::kPADDING:
        case nvinfer1::LayerType::kSHUFFLE:
        case nvinfer1::LayerType::kREDUCE:
        case nvinfer1::LayerType::kTOPK:
        case nvinfer1::LayerType::kGATHER:
        case nvinfer1::LayerType::kMATRIX_MULTIPLY:
        case nvinfer1::LayerType::kRAGGED_SOFTMAX:
        case nvinfer1::LayerType::kCONSTANT:
        case nvinfer1::LayerType::kRNN_V2:
        case nvinfer1::LayerType::kIDENTITY:
        case nvinfer1::LayerType::kPLUGIN_V2:
        case nvinfer1::LayerType::kSLICE:
        case nvinfer1::LayerType::kSHAPE:
        case nvinfer1::LayerType::kPARAMETRIC_RELU:
        case nvinfer1::LayerType::kRESIZE:
            return "";
        }
        return "Unknow layer type";
    }

    static bool layer_has_input_tensor(nvinfer1::ILayer *layer)
    {
        int num_input = layer->getNbInputs();
        for (int i = 0; i < num_input; ++i)
        {
            auto input = layer->getInput(i);
            if (input == nullptr)
                continue;

            if (input->isNetworkInput())
                return true;
        }
        return false;
    }

    static bool layer_has_output_tensor(nvinfer1::ILayer *layer)
    {
        int num_output = layer->getNbOutputs();
        for (int i = 0; i < num_output; ++i)
        {

            auto output = layer->getOutput(i);
            if (output == nullptr)
                continue;

            if (output->isNetworkOutput())
                return true;
        }
        return false;
    }

    template <typename _T>
    std::shared_ptr<_T> make_nvshared(_T *ptr)
    {
        return std::shared_ptr<_T>(ptr, [](_T *p)
                                   { p->destroy(); });
    }

    const char *mode_string(Mode type)
    {
        switch (type)
        {
        case Mode::FP32:
            return "FP32";
        case Mode::FP16:
            return "FP16";
        default:
            return "UnknowTRTMode";
        }
    }

    static nvinfer1::Dims convert_to_trt_dims(const std::vector<int> &dims)
    {

        nvinfer1::Dims output{0};
        if (dims.size() > nvinfer1::Dims::MAX_DIMS)
        {
            INFOE("convert failed, dims.size[%d] > MAX_DIMS[%d]", dims.size(), nvinfer1::Dims::MAX_DIMS);
            return output;
        }

        if (!dims.empty())
        {
            output.nbDims = dims.size();
            memcpy(output.d, dims.data(), dims.size() * sizeof(int));
        }
        return output;
    }

    static std::string align_blank(const std::string &input, int align_size, char blank = ' ')
    {
        if (input.size() >= align_size)
            return input;
        std::string output = input;
        for (int i = 0; i < align_size - input.size(); ++i)
            output.push_back(blank);
        return output;
    }

    static long long timestamp_now()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static double timestamp_now_float()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    }

    bool compile(
        Mode mode,
        unsigned int maxBatchSize,
        const std::string &source,
        const std::string &saveto,
        const size_t maxWorkspaceSize)
    {

        INFO("Compile %s %s.", mode_string(mode), source.c_str());
        auto builder = make_nvshared(nvinfer1::createInferBuilder(gLogger));
        if (builder == nullptr)
        {
            INFOE("Can not create builder.");
            return false;
        }

        auto config = make_nvshared(builder->createBuilderConfig());
        if (mode == Mode::FP16)
        {
            if (!builder->platformHasFastFp16())
            {
                INFOW("Platform not have fast fp16 support.");
            }
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        std::shared_ptr<nvinfer1::INetworkDefinition> network;
        // shared_ptr<ICaffeParser> caffeParser;
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network = make_nvshared(builder->createNetworkV2(explicitBatch));

        std::shared_ptr<nvonnxparser::IParser> onnxParser = make_nvshared(nvonnxparser::createParser(*network, gLogger));
        if (onnxParser == nullptr)
        {
            INFOE("Can not create parser.");
            return false;
        }

        if (!onnxParser->parseFromFile(source.c_str(), 1))
        {
            INFOE("Can not parse OnnX file: %s", source.c_str());
            return false;
        }

        auto inputTensor = network->getInput(0);
        auto inputDims = inputTensor->getDimensions();

        INFO("Input shape is %s", join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
        INFO("Set max batch size = %d", maxBatchSize);
        INFO("Set max workspace size = %.2f MB", maxWorkspaceSize / 1024.0f / 1024.0f);
        INFO("Base device: %s", CUDATools::device_description().c_str());

        int net_num_input = network->getNbInputs();
        INFO("Network has %d inputs:", net_num_input);
        std::vector<std::string> input_names(net_num_input);
        for (int i = 0; i < net_num_input; ++i)
        {
            auto tensor = network->getInput(i);
            auto dims = tensor->getDimensions();
            auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
            INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

            input_names[i] = tensor->getName();
        }

        int net_num_output = network->getNbOutputs();
        INFO("Network has %d outputs:", net_num_output);
        for (int i = 0; i < net_num_output; ++i)
        {
            auto tensor = network->getOutput(i);
            auto dims = tensor->getDimensions();
            auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
            INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
        }

        int net_num_layers = network->getNbLayers();
        INFO("Network has %d layers:", net_num_layers);
        for (int i = 0; i < net_num_layers; ++i)
        {
            auto layer = network->getLayer(i);
            auto name = layer->getName();
            auto type_str = layer_type_name(layer);
            auto input0 = layer->getInput(0);
            if (input0 == nullptr)
                continue;

            auto output0 = layer->getOutput(0);
            auto input_dims = input0->getDimensions();
            auto output_dims = output0->getDimensions();
            bool has_input = layer_has_input_tensor(layer);
            bool has_output = layer_has_output_tensor(layer);
            auto descript = layer_descript(layer);
            type_str = align_blank(type_str, 18);
            auto input_dims_str = align_blank(dims_str(input_dims), 18);
            auto output_dims_str = align_blank(dims_str(output_dims), 18);
            auto number_str = align_blank(format("%d.", i), 4);

            const char *token = "      ";
            if (has_input)
                token = "  >>> ";
            else if (has_output)
                token = "  *** ";

            INFOV("%s%s%s %s-> %s%s", token,
                  number_str.c_str(),
                  type_str.c_str(),
                  input_dims_str.c_str(),
                  output_dims_str.c_str(),
                  descript.c_str());
        }

        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(maxWorkspaceSize);

        auto profile = builder->createOptimizationProfile();
        for (int i = 0; i < net_num_input; ++i)
        {
            auto input = network->getInput(i);
            auto input_dims = input->getDimensions();
            input_dims.d[0] = 1;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
            input_dims.d[0] = maxBatchSize;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        }

        // not need
        // for(int i = 0; i < net_num_output; ++i){
        // 	auto output = network->getOutput(i);
        // 	auto output_dims = output->getDimensions();
        // 	output_dims.d[0] = 1;
        // 	profile->setDimensions(output->getName(), nvinfer1::OptProfileSelector::kMIN, output_dims);
        // 	profile->setDimensions(output->getName(), nvinfer1::OptProfileSelector::kOPT, output_dims);
        // 	output_dims.d[0] = maxBatchSize;
        // 	profile->setDimensions(output->getName(), nvinfer1::OptProfileSelector::kMAX, output_dims);
        // }
        config->addOptimizationProfile(profile);

        // error on jetson
        // auto timing_cache = shared_ptr<nvinfer1::ITimingCache>(config->createTimingCache(nullptr, 0), [](nvinfer1::ITimingCache* ptr){ptr->reset();});
        // config->setTimingCache(*timing_cache, false);
        // config->setFlag(BuilderFlag::kGPU_FALLBACK);
        // config->setDefaultDeviceType(DeviceType::kDLA);
        // config->setDLACore(0);

        INFO("Building engine...");
        auto time_start = timestamp_now();
        auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
        if (engine == nullptr)
        {
            INFOE("engine is nullptr");
            return false;
        }
        INFO("Build done %lld ms !", timestamp_now() - time_start);

        // serialize the engine, then close everything down
        auto seridata = make_nvshared(engine->serialize());
        return save_file(saveto, seridata->data(), seridata->size());
    }
}; // namespace TRT