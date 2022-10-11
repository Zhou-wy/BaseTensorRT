/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-04 20:30:45
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-08 21:10:56
 */
#include <iostream>
#include "yolox.h"
int main(int argc, char const *argv[])
{
    TRTLogger logger;
    cv::VideoCapture capture;
    cv::Mat frame;
    frame = capture.open("../workspace/video3.mp4");
    cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    writer.open("../workspace/video3-output.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size, true);
    if (!capture.isOpened())
    {
        printf("can not open video!\n");
        return -1;
    }
    if (!build_model("../workspace/yolox.trtmodel", "../workspace/mm-yolox.onnx", logger))
        return -1;
    while (capture.read(frame))
    {
        cv::Mat det = inference(frame, "../workspace/yolox.trtmodel", logger);
        writer.write(det);
        cv::waitKey(10);
    }
    capture.release();
    return 0;
}
