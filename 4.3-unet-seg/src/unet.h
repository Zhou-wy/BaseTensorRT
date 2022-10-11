/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-05 09:49:57
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-05 15:07:13
 */
#ifndef __UNET_SEG_H
#define __UNET_SEG_H

// System include
#include <iostream>
#include <fstream>
#include <chrono>
#include <unistd.h>

// TensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

// CUDA include
#include <cuda.h>
#include <cuda_runtime.h>

// OpenCV include
#include <opencv2/opencv.hpp>

//////////////////////////////////////////////////////////////////////////////
bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__);

static std::vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12};

inline const char *severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "[internal_error]";
    case nvinfer1::ILogger::Severity::kERROR:
        return "[error]";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "[warning]";
    case nvinfer1::ILogger::Severity::kINFO:
        return "[info]";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "[verbose]";
    default:
        return "[unknow]";
    }
}

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

bool build_model(const std::string &trt_model_file, const std::string &onnx_model_file, TRTLogger &logger);

void inference(cv::Mat &image, const std::string &trt_model_file, const std::string &save_result_image, TRTLogger &logger);

#endif