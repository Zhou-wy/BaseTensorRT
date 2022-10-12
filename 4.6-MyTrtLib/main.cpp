/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-09 13:35:16
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-12 15:03:41
 */

// tensorRT include
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

// personal include
#include "src/utils/cuda_tools.cuh"
#include "src/utils/simple_logger.hpp"
#include "src/builder/trt_builder.hpp"
#include "src/memory/mix_memory.hpp"
#include "src/Tensor/Trt_Tensor.hpp"
#include "src/infer/Trt_Infer.hpp"

bool exists(const std::string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
bool build_model()
{

    if (exists("../workspace/model.engine"))
    {
        printf("model.engine has exists.\n");
        return true;
    }

    // SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        10,
        "../workspace/classifier.onnx",
        "../workspace/model.engine",
        1 << 28);
    INFO("Done.");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::string> load_labels(const char *file)
{
    std::vector<std::string> lines;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        printf("open %s failed.\n", file);
        return lines;
    }

    std::string line;
    while (getline(in, line))
    {
        lines.push_back(line);
    }
    in.close();
    return lines;
}

void inference()
{

    auto engine = TRT::load_infer("../workspace/model.engine");
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        return;
    }

    engine->print();

    auto input = engine->input(0);
    auto output = engine->output(0);
    int input_width = input->width();
    int input_height = input->height();

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("../workspace/dog.jpg");
    float mean[] = {0.406, 0.456, 0.485};
    float std[] = {0.225, 0.224, 0.229};

    // 对应于pytorch的代码部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    image.convertTo(image, CV_32F);

    cv::Mat channel_based[3];
    for (int i = 0; i < 3; ++i)
        channel_based[i] = cv::Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2 - i));

    cv::split(image, channel_based);
    for (int i = 0; i < 3; ++i)
        channel_based[i] = (channel_based[i] / 255.0f - mean[i]) / std[i];

    engine->forward(true);

    int num_classes = output->size(1);
    float *prob = output->cpu<float>();
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels = load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);
}

int main()
{
    if (!build_model())
    {
        return -1;
    }
    inference();
    return 0;
}