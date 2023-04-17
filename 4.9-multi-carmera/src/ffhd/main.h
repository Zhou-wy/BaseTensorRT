/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-12-02 09:35:42
 * @LastEditors: zwy
 * @LastEditTime: 2022-12-06 19:06:38
 */
//
//  main.h

//  Created by hengfeng zhuo on 2019/7/20.
//  Copyright © 2019 hengfeng zhuo. All rights reserved.
//

#ifndef main_h
#define main_h

#include <stdio.h>
#include <istream>
#include <iostream>
#include <inttypes.h>
#include <string>

/// 1. FFMpeg头文件
extern "C"
{

#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libpostproc/postprocess.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/avassert.h>
#include <libavutil/avstring.h>
#include <libavutil/bprint.h>
#include <libavutil/display.h>
#include <libavutil/mathematics.h>
#include <libavutil/imgutils.h>
//#include <libavutil/libm.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/eval.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/cpu.h>
#include <libavutil/ffversion.h>
#include <libavutil/version.h>
}

/// 2. OpenCV头文件
#include <opencv2/core.hpp>    // Mat
#include <opencv2/highgui.hpp> // 测试用的GUI
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/// 3. QT5头文件
#include <QApplication>
#include <QtCore/QCoreApplication>
#include <QAudioInput>
#include <QDebug>
#include <QMainWindow>
#include <QListWidget>
#include <QListWidgetItem>
#include <QThread>

#endif /* main_h */
