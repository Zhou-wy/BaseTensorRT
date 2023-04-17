/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-12-05 13:26:16
 * @LastEditors: zwy
 * @LastEditTime: 2022-12-12 14:14:09
 */
// tensorRT include
#include <NvInfer.h>
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

#include "../src/TrtLib/common/ilogger.hpp"
#include "../src/TrtLib/builder/trt_builder.hpp"
#include "../src/app_yolo/yolo.hpp"
#include "../src/app_http/http_server.hpp"

#include "../src/ffhd/XVideoCapture.h"
#include "../src/ffhd/XMediaEncode.h"
#include "../src/ffhd/XRtmp.h"
#include "../src/ffhd/XAudioRecord.h"
#include "../src/ffhd/XVideoCapture.h"

using namespace std;

const string engine_file = "../workspace/yolov5s.engine";
const string onnx_file = "../workspace/yolov5s.onnx";
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

const int color_list[80][3] =
    {
        //{255 ,255 ,255}, //bg
        {216, 82, 24},
        {236, 176, 31},
        {125, 46, 141},
        {118, 171, 47},
        {76, 189, 237},
        {238, 19, 46},
        {76, 76, 76},
        {153, 153, 153},
        {255, 0, 0},
        {255, 127, 0},
        {190, 190, 0},
        {0, 255, 0},
        {0, 0, 255},
        {170, 0, 255},
        {84, 84, 0},
        {84, 170, 0},
        {84, 255, 0},
        {170, 84, 0},
        {170, 170, 0},
        {170, 255, 0},
        {255, 84, 0},
        {255, 170, 0},
        {255, 255, 0},
        {0, 84, 127},
        {0, 170, 127},
        {0, 255, 127},
        {84, 0, 127},
        {84, 84, 127},
        {84, 170, 127},
        {84, 255, 127},
        {170, 0, 127},
        {170, 84, 127},
        {170, 170, 127},
        {170, 255, 127},
        {255, 0, 127},
        {255, 84, 127},
        {255, 170, 127},
        {255, 255, 127},
        {0, 84, 255},
        {0, 170, 255},
        {0, 255, 255},
        {84, 0, 255},
        {84, 84, 255},
        {84, 170, 255},
        {84, 255, 255},
        {170, 0, 255},
        {170, 84, 255},
        {170, 170, 255},
        {170, 255, 255},
        {255, 0, 255},
        {255, 84, 255},
        {255, 170, 255},
        {42, 0, 0},
        {84, 0, 0},
        {127, 0, 0},
        {170, 0, 0},
        {212, 0, 0},
        {255, 0, 0},
        {0, 42, 0},
        {0, 84, 0},
        {0, 127, 0},
        {0, 170, 0},
        {0, 212, 0},
        {0, 255, 0},
        {0, 0, 42},
        {0, 0, 84},
        {0, 0, 127},
        {0, 0, 170},
        {0, 0, 212},
        {0, 0, 255},
        {0, 0, 0},
        {36, 36, 36},
        {72, 72, 72},
        {109, 109, 109},
        {145, 145, 145},
        {182, 182, 182},
        {218, 218, 218},
        {0, 113, 188},
        {80, 182, 188},
        {127, 127, 0},
};

int main(int argc, char const *argv[])
{
    INFO("-------------------begin--------------------------");
    Yolo::BoxArray bbox;
    iLogger::set_logger_save_directory("/home/zwy/CWorkspace/BaseTensorRT/4.9-multi-carmera/workspace/log");
    const char *inUrl = "rtsp://admin:admin123@192.168.0.213:554/cam/realmonitor?channel=1&subtype=0";
    const char *outUrl = "rtmp://192.168.0.7/live/test";
    INFO("%s", outUrl);
    XVideoCapture *xv = XVideoCapture::Get();
    if (!xv->Init(inUrl))
    {
        INFOE("open camera failed!");
        getchar();
        return -1;
    }
    INFO("open camera success!");
    xv->Start();

    /// 音视频编码类
    XMediaEncode *xe = XMediaEncode::getInstance();

    /// 2 初始化格式转换上下文
    /// 3 初始化输出的数据结构
    xe->inWidth = xv->width;
    xe->inHeight = xv->height;
    xe->outWidth = xv->width;
    xe->outHeight = xv->height;
    if (!xe->InitScale())
    {
        getchar();
        return -1;
    }
    INFO("初始化视频像素转换上下文成功!");

    /// 初始化视频编码器
    if (!xe->InitVideoCodec())
    {
        getchar();
        return -1;
    }

    /// 5 输出封装器和音频流配置
    // a 创建输出封装器上下文
    XRtmp *xr = XRtmp::getInstance(0);
    if (!xr->InitMux(outUrl))
    {
        getchar();
        return -1;
    }

    // 添加视频流
    int vindex = 0;
    vindex = xr->AddStream(xe->videoCodecContext);
    if (vindex < 0)
    {
        getchar();
        return -1;
    }

    /// 打开rtmp 的网络输出IO
    // 写入封装头
    if (!xr->SendMuxHead())
    {
        getchar();
        return -1;
    }

    for (;;)
    {
        XData vd = xv->Pop();
        // 处理视频
        if (vd.size > 0)
        {
            AVFrame *yuv = xe->RGBToYUV((unsigned char *)(vd.data));
            vd.Drop();
            AVPacket *pkt = xe->EncodeVideo(yuv);
            if (pkt)
            {
                // 推流
                if (!xr->SendFrame(pkt, vindex))
                {
                    cout << "推流失败" << flush;
                }
            }
        }
    }
    INFO("-------------------over--------------------------");
    return 0;
}
