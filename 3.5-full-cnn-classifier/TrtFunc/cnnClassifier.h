/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-03 15:49:53
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-03 18:41:59
 */
#ifndef __CNN_CLASSIFIER_H
#define __CNN_CLASSIFIER_H
// TensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

// CUDA include
#include <cuda.h>
#include <cuda_runtime.h>
// OpenCV include
#include <opencv2/opencv.hpp>
// system include
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <cmath>

#include <unistd.h>
#include <stdio.h>
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__);
const char *severity_string(nvinfer1::ILogger::Severity t);
class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kINFO)
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
            if (severity == nvinfer1::ILogger::Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= nvinfer1::ILogger::Severity::kERROR)
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
bool build_model(const std::string &onnx_model_path, const std::string &trt_model_path, TRTLogger &logger);
void inference(const cv::String &image_filename, const char *label_file, const std::string &trt_model_file, TRTLogger loggero);
#endif