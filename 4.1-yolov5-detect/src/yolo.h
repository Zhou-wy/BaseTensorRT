/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-04 12:17:56
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-04 16:00:49
 */
#ifndef _YOLO_DETECT_H
#define _YOLO_DETECT_H

// system include
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <stdio.h>
#include <math.h>

// TensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

// CUDA include
#include <cuda.h>
#include <cuda_runtime.h>

// OpenCV include
#include <opencv2/opencv.hpp>

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__);

// coco数据集的labels，关于coco：https://cocodataset.org/#home
static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

const char *severity_string(nvinfer1::ILogger::Severity t);

class TRTLogger : public nvinfer1::ILogger
{

public:
    virtual void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
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

bool build_model(const std::string &onnx_model_path, const std::string &trt_model_path, TRTLogger &logger);

bool inference(cv::Mat &src_image, const std::string &trt_model_path, const std::string &save_output_image_path, TRTLogger &logger, bool save_input_image);
#endif
