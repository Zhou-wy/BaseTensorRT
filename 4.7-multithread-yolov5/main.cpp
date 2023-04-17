/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-12 16:06:24
 * @LastEditors: zwy
 * @LastEditTime: 2023-02-14 14:31:56
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
#include <chrono>

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

static const char *labels[] = {
    "bost",
    "ship",
    "ball",
    "bridge",
    "rock",
    "person",
    "rubbish",
    "mast",
    "buoy",
    "platfrom",
    "harbor",
    "tree",
    "grass",
    "animal",
};

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

    if (exists("/home/zwy/CWorkspace/BaseTensorRT/4.7-multithread-yolov5/workspace/best.engine"))
    {
        printf("../workspace/best.engine has exists.\n");
        return true;
    }

    // SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        10,
        "/home/zwy/CWorkspace/BaseTensorRT/4.7-multithread-yolov5/workspace/best.onnx",
        "/home/zwy/CWorkspace/BaseTensorRT/4.7-multithread-yolov5/workspace/best.engine",
        1 << 28);
    INFO("Done.");
    return true;
}

void inference_video()
{
    cv::Mat image, output;
    cout << CV_VERSION << endl;
    srand((unsigned)time(NULL));

    const string videoStreamAddress = "rtsp://admin:admin123@192.168.0.213:554/cam/realmonitor?channel=1&subtype=0";
    /*
    格式说明：
    1、username: 设备登录用户名。例如admin。`
    2、password: 设备登录密码。例如admin123。
    3、ip: 设备IP地址。例如192.168.1.108
    4、port: 端口号默认为554，若为默认可不填写。
    5、channel: 通道号，起始为1。例如通道2，则为channel=2。
    6、subtype: 码流类型，主码流为0（即subtype=0），辅码流为1（即subtype=1）。
    */

    cv::VideoCapture cap = cv::VideoCapture(videoStreamAddress);

    auto yolov5 = YoloV5::create_infer("../workspace/best.engine");
    cv::Size size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    char Fps[32];

    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return;
    }
    else
    {
        cout << "success" << endl;
    }

    while (cap.read(image))
    {
        auto t = cv::getTickCount();
        auto boxes = yolov5->commit(image).get();

        for (auto &box : boxes)
        {
            cv::Scalar color(0, 255, 0);
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

            auto name = cocolabels[box.class_label];
            auto caption = cv::format("%s %.2f", name, box.confidence);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

            cv::rectangle(image, cv::Point(box.left - 3, box.top - 33), cv::Point(box.left + text_width, box.top), color, -1);
            cv::putText(image, caption, cv::Point(box.left, box.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        auto d = cv::getTickCount();
        // double spendTime = (d - t) * 1000 / cv::getTickFrequency();
        double fps = cv::getTickFrequency() / (d - t);

        auto fpsString = cv::format("%s %.2f", "FPS:", fps);
        std::cout << fpsString << std::endl;
        cv::putText(image, fpsString, cv::Point(5, 20), 0, 1, cv::Scalar::all(0), 2, 16); // 字体颜色

        cv::namedWindow("Output Window");
        cv::imshow("Output Window", image);
        if (cv::waitKey(1) >= 0)
            break;
    }
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
    auto yolov5 = YoloV5::create_infer("../workspace/best.engine");
    cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    char Fps[32];
    double t;
    writer.open("../workspace/video-output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 10, size, true);
    while (capture.read(frame))
    {
        t = (double)cv::getTickCount();

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
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        double fps = 1.0 / t;

        auto fpsString = cv::format("%s %.2f", "FPS:", fps);
        std::cout << fpsString << std::endl;
        cv::putText(frame, fpsString, cv::Point(5, 20), 0, 1, cv::Scalar::all(0), 2, 16); // 字体颜色

        writer.write(frame);
        cv::waitKey(10);
    }

    // cv::imwrite("../workspace/image-draw.jpg", image);
}

void inference_1()
{
    auto yolov5 = YoloV5::create_infer("/home/zwy/CWorkspace/BaseTensorRT/4.7-multithread-yolov5/workspace/best.engine");
    auto src = cv::imread("/home/zwy/CWorkspace/BaseTensorRT/4.7-multithread-yolov5/workspace/20.jpg");
    auto boxes = yolov5->commit(src).get();

    for (auto &box : boxes)
    {
        cv::Scalar color(0, 255, 0);
        cv::rectangle(src, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

        auto name = labels[box.class_label];
        auto caption = cv::format("%s %.2f", name, box.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;

        cv::rectangle(src, cv::Point(box.left - 3, box.top - 33), cv::Point(box.left + text_width, box.top), color, -1);
        cv::putText(src, caption, cv::Point(box.left, box.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("/home/zwy/CWorkspace/BaseTensorRT/4.7-multithread-yolov5/workspace/image-draw-src.jpg", src);
}

int main(int argc, char const *argv[])
{
    if (!build_model())
    {
        return -1;
    }
    // inference_video();
    inference_1();
    return 0;
}
