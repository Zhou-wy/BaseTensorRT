/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-13 09:13:25
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-15 09:18:57
 */

// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

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

#include "src/TrtLib/common/ilogger.hpp"
#include "src/TrtLib/builder/trt_builder.hpp"
#include "src/app_yolo/yolo.hpp"
#include "src/app_http/http_server.hpp"

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

class LogicalController : public Controller
{
    SetupController(LogicalController);

public:
    bool startup();

public:
    DefRequestMapping(detect);

public:
    DefRequestMapping(getCustom);
    DefRequestMapping(getReturn);
    DefRequestMapping(getBinary);
    DefRequestMapping(getFile);
    DefRequestMapping(putBase64Image);

private:
    shared_ptr<Yolo::Infer> yolo_;
};

bool LogicalController::startup()
{
    yolo_ = Yolo::create_infer("../workspace/yolov5s.engine", Yolo::Type::V5, 0, 0.25, 0.45);
    return yolo_ != nullptr;
}

Json::Value LogicalController::detect(const Json::Value &param)
{

    auto session = get_current_session();
    if (session->request.body.empty())
        return failure("Request body is empty");

    // if base64
    // iLogger::base64_decode();
    cv::Mat imdata(1, session->request.body.size(), CV_8U, (char *)session->request.body.data());
    cv::Mat image = cv::imdecode(imdata, 1);
    if (image.empty())
        return failure("Image decode failed");

    auto boxes = yolo_->commit(image).get();
    Json::Value out(Json::arrayValue);
    for (int i = 0; i < boxes.size(); ++i)
    {
        auto &item = boxes[i];
        Json::Value itemj;
        itemj["left"] = item.left;
        itemj["top"] = item.top;
        itemj["right"] = item.right;
        itemj["bottom"] = item.bottom;
        itemj["class_label"] = item.class_label;
        itemj["confidence"] = item.confidence;
        out.append(itemj);
    }
    return success(out);
}
Json::Value LogicalController::putBase64Image(const Json::Value &param)
{

    /**
     * 注意，这个函数的调用，请用工具（postman）以提交body的方式(raw)提交base64数据
     * 才能够在request.body中拿到对应的base64，并正确解码后储存
     * 1. 可以在网页上提交一个图片文件，并使用postman进行body-raw提交，例如网址是：https://base64.us/，选择页面下面的“选择文件”按钮
     * 2. 去掉生成的base64数据前缀：data:image/png;base64,。保证是纯base64数据输入
     *   这是一个图像的base64案例：iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsQAAA7EAZUrDhsAAABLSURBVEhLY2RY9OI/Ax0BE5SmG6DIh/8DJKAswoBxwwswTXcfjlpIdTBqIdXBqIVUB8O/8B61kOpg1EKqg1ELqQ5GLaQ6oLOFDAwA5z0K0dyTzgcAAAAASUVORK5CYII=
     *   提交后能看到是个天蓝色的背景加上右上角有黄色的正方形
     */

    auto session = get_current_session();
    auto image_data = iLogger::base64_decode(session->request.body);
    iLogger::save_file("base_decode.jpg", image_data);
    return success();
}

Json::Value LogicalController::getCustom(const Json::Value &param)
{

    auto session = get_current_session();
    const char *output = "hello http server";
    session->response.write_binary(output, strlen(output));
    session->response.set_header("Content-Type", "text/plain");
    return success();
}

Json::Value LogicalController::getReturn(const Json::Value &param)
{

    Json::Value data;
    data["alpha"] = 3.15;
    data["beta"] = 2.56;
    data["name"] = "张三";
    return success(data);
}

Json::Value LogicalController::getBinary(const Json::Value &param)
{

    auto session = get_current_session();
    auto data = iLogger::load_file("img.jpg");
    session->response.write_binary(data.data(), data.size());
    session->response.set_header("Content-Type", "image/jpeg");
    return success();
}

Json::Value LogicalController::getFile(const Json::Value &param)
{

    auto session = get_current_session();
    session->response.write_file("img.jpg");
    return success();
}

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
        printf("../workspace/yolov5s.engine has exists.\n");
        return true;
    }

    // SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    TRT::compile(
        TRT::Mode::FP32,
        10,
        "../workspace/yolov5s.onnx",
        "../workspace/yolov5s.engine");
    INFO("Done.");
    return true;
}

int start_http(int port = 8090)
{

    INFO("Create controller");
    auto logical_controller = make_shared<LogicalController>();
    if (!logical_controller->startup())
    {
        INFOE("Startup controller failed.");
        return -1;
    }

    string address = iLogger::format("0.0.0.0:%d", port);
    INFO("Create http server to: %s", address.c_str());

    auto server = createHttpServer(address, 32);
    if (!server)
        return -1;

    server->verbose();

    INFO("Add controller");
    server->add_controller("/api", logical_controller);

    // 这是一个vue的项目
    server->add_controller("/", create_redirect_access_controller("./web"));
    server->add_controller("/static", create_file_access_controller("./"));
    INFO("Access url: http://%s", address.c_str());

    INFO(
        "\n"
        "访问如下地址即可看到效果:\n"
        "1. http://%s/api/getCustom              使用自定义写出内容作为response\n"
        "2. http://%s/api/getReturn              使用函数返回值中的json作为response\n"
        "3. http://%s/api/getBinary              使用自定义写出二进制数据作为response\n"
        "4. http://%s/api/getFile                使用自定义写出文件路径作为response\n"
        "5. http://%s/api/putBase64Image         通过提交base64图像数据进行解码后储存\n"
        "6. http://%s/static/img.jpg             直接访问静态文件处理的controller,具体请看函数说明\n"
        "7. http://%s/api/detect                 访问web页面,vue开发的",

        address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str());

    INFO("按下Ctrl + C结束程序");
    return iLogger::while_loop();
}

int main()
{
    iLogger::set_log_level(iLogger::LogLevel::Debug);
    iLogger::set_logger_save_directory("../workspace/log");
    // 新的实现
    if (!build_model())
    {
        return -1;
    }
    return start_http();
}