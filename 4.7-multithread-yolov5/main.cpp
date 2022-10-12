/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-12 16:06:24
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-12 19:16:16
 */

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <NvOnnxParser.h>

// 推理用的运行时头文件
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

#include "src/TensorRT/builder/trt_builder.hpp"
#include "src/TensorRT/utils/simple_logger.hpp"
#include "src/app_yolo/yolov5.hpp"

using namespace std;

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

static bool exists(const string &path)
{

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
static bool build_model()
{

    if (exists("../workspace/yolov5s.engine"))
    {
        printf("workspace/yolov5s.engine has exists.\n");
        return true;
    }

    // SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        10,
        "../workspace/yolov5s.onnx",
        "../workspace/yolov5s.engine",
        1 << 28);
    INFO("Done.");
    return true;
}

static void inference()
{
    cv::VideoCapture capture;
    cv::Mat frame;
    frame = capture.open("../workspace/video.mp4");
    if (!capture.isOpened())
    {
        printf("can not open video!\n");
        return;
    }
    // auto image = cv::imread("../workspace/rq.jpg");
    auto yolov5 = YoloV5::create_infer("../workspace/yolov5s.engine");
    cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    writer.open("../workspace/video-output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 10, size, true);
    while (capture.read(frame))
    {
        auto boxes = yolov5->commit(frame).get();
        for (auto &box : boxes)
        {
            cv::Scalar color(0, 255, 0);
            cv::rectangle(frame, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

            auto name = cocolabels[box.class_label];
            auto caption = cv::format("%s %.2f", name, box.confidence);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(box.left - 3, box.top - 33), cv::Point(box.left + text_width, box.top), color, -1);
            cv::putText(frame, caption, cv::Point(box.left, box.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        writer.write(frame);
        cv::waitKey(10);
    }

    // cv::imwrite("../workspace/image-draw.jpg", image);
}

int main(int argc, char const *argv[])
{
    if (!build_model())
    {
        return -1;
    }
    inference();
    return 0;
}
