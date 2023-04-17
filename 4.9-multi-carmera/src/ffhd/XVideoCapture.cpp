//
// Created by hengfeng zhuo on 2019-08-05.
//

#include <opencv2/highgui.hpp>
#include "XVideoCapture.h"
#include "main.h"
#include "../TrtLib/common/ilogger.hpp"
#include "../app_yolo/yolo.hpp"
#include "../TrtLib/builder/trt_builder.hpp"
using namespace cv;


class CXVideoCapture : public XVideoCapture
{

public:
    /**
     * 打开本地摄像头
     * @param camIndex
     * @return
     */
    bool Init(int camIndex = 0) override
    {
        // 打开本地摄像头
        cam.open(camIndex);
        if (!cam.isOpened())
        {
            INFOE("打开摄像头失败");
            return true;
        }

        // 得到本地相机参数
        this->width = cam.get(CAP_PROP_FRAME_WIDTH);
        this->height = cam.get(CAP_PROP_FRAME_HEIGHT);
        this->fps = cam.get(CAP_PROP_FPS);
        INFO("打开摄像头成功 width= %d, height=%d, fps= %d", width, height, fps);
        if (fps <= 0)
        {
            fps = 25;
        }
        return true;
    }

    /**
     * 打开流媒体
     * @param url
     * @return
     */
    bool Init(const char *url) override
    {
        cam.open(url);
        if (!cam.isOpened())
        {
            INFOE("打开流媒体失败:%s", url);
            return true;
        }

        // 得到流媒体的参数
        this->width = cam.get(CAP_PROP_FRAME_WIDTH);
        this->height = cam.get(CAP_PROP_FRAME_HEIGHT);
        this->fps = cam.get(CAP_PROP_FPS);
        Yolo::BoxArray bbox;
        INFO("打开流媒体成功 width= %d, height=%d, fps= %d", width, height, fps);
        if (fps <= 0)
        {
            fps = 25;
        }
        cam.read(this->frame);
        return true;
    }

    void Stop() override
    {
        // 停止线程
        XDataThread::Stop();
        // 是否camera
        if (cam.isOpened())
        {
            cam.release();
        }
    }
    // cv::Mat GetFrame() override
    // {
    //     return this->frame;
    // }

protected:
    VideoCapture cam; // 本地摄像头
    cv::Mat frame;
    /**
     * 这个是QThread的线程执行体
     */
    void
    run() override
    {
        INFO("开始执行video capture线程");

        // Mat frame;
        while (!isExit)
        {
            // 读取一帧
            if (!cam.read(this->frame) || this->frame.empty())
            {
                msleep(1); // 如果没有读取到，等待1ms
                continue;
            }
            // 生成一个XData
            int size = this->frame.rows * this->frame.cols * this->frame.elemSize();
            // std::cout << "得到一帧数据, 大小 = " << frame.rows << " * "<< frame.cols << std::endl;
            XData d((char *)this->frame.data, size);
            Push(d);
        }
        INFO("退出video capture线程");
    }
};

XVideoCapture *XVideoCapture::Get(unsigned char index)
{
    static CXVideoCapture xc[255];
    return &xc[0];
}

XVideoCapture::XVideoCapture()
{
}

XVideoCapture::~XVideoCapture()
{
}
