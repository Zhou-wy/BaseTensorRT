/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-12 16:09:06
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-12 16:47:39
 */
#ifndef YOLOV5_HPP
#define YOLOV5_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <future>
#include <opencv2/opencv.hpp>

namespace YoloV5
{
    struct Box
    {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };
    typedef std::vector<Box> BoxArray;

    class Infer
    {
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat &input) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string &file,
        int gpuid = 0, float confidence_threshold = 0.25, float nms_threshold = 0.45);
};

#endif
