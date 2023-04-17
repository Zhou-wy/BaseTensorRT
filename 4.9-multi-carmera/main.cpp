/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-13 09:13:25
 * @LastEditors: zwy
 * @LastEditTime: 2022-11-13 16:47:42
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

#include "src/TrtLib/common/ilogger.hpp"
#include "src/TrtLib/builder/trt_builder.hpp"
#include "src/app_yolo/yolo.hpp"
#include "src/app_http/http_server.hpp"

using namespace std;

const string engine_file = "workspace/yolov5s.engine";
const string onnx_file = "workspace/yolov5s.onnx";
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

class InferYolo
{
public:
    InferYolo() = delete;
    InferYolo(std::string onnxFile, std::string engineFile) : m_onnxFile(onnx_file), m_engineFile(engineFile)
    {
        m_yoloIns = get_infer(Yolo::Type::V5);
    };

    bool inference(const cv::Mat &image_input, Yolo::BoxArray &boxarray)
    {

        if (m_yoloIns == nullptr)
        {
            INFOE("Not Initialize.");
            return false;
        }

        if (image_input.empty())
        {
            INFOE("Image is empty.");
            return false;
        }
        boxarray = m_yoloIns->commit(image_input).get();
        return true;
    }

private:
    shared_ptr<Yolo::Infer> get_infer(Yolo::Type type)
    {
        if (!iLogger::exists(m_engineFile))
        {
            TRT::compile(
                TRT::Mode::FP32,
                10,
                m_onnxFile,
                m_engineFile);
        }
        else
        {
            INFOW("%s has been created!", m_engineFile.c_str());
        }
        return Yolo::create_infer(m_engineFile, type, 0, 0.25, 0.45);
    }
    shared_ptr<Yolo::Infer> m_yoloIns;
    std::string m_engineFile;
    std::string m_onnxFile;
};

class InferInstance
{
public:
    bool startup()
    {
        yoloIns = get_infer(Yolo::Type::V5);
        return yoloIns != nullptr;
    }

    bool inference(const cv::Mat &image_input, Yolo::BoxArray &boxarray)
    {

        if (yoloIns == nullptr)
        {
            INFOE("Not Initialize.");
            return false;
        }

        if (image_input.empty())
        {
            INFOE("Image is empty.");
            return false;
        }
        boxarray = yoloIns->commit(image_input).get();
        return true;
    }

private:
    shared_ptr<Yolo::Infer> get_infer(Yolo::Type type)
    {
        if (!iLogger::exists(engine_file))
        {
            TRT::compile(
                TRT::Mode::FP32,
                10,
                onnx_file,
                engine_file);
        }
        else
        {
            INFOW("%s has been created!", engine_file.c_str());
        }
        return Yolo::create_infer(engine_file, type, 0, 0.25, 0.45);
    }
    shared_ptr<Yolo::Infer> yoloIns;
};

class LogicalController : public Controller
{
    SetupController(LogicalController);

public:
    bool startup();

public:
    DefRequestMapping(detect);
    DefRequestMapping(getCustom);
    DefRequestMapping(getReturn);
    DefRequestMapping(getBinary);
    DefRequestMapping(getFile);
    DefRequestMapping(putBase64Image);
    DefRequestMapping(detectBase64Image);

private:
    shared_ptr<InferInstance> infer_instance_;
};

/// ######################################
Json::Value LogicalController::detect(const Json::Value &param)
{
    auto session = get_current_session();
    auto image_data = iLogger::base64_decode(session->request.body);
    if (image_data.empty())
        return failure("Image is required");

    auto image = cv::imdecode(image_data, 1);
    if (image_data.empty())
        return failure("Image is empty");

    Yolo::BoxArray boxarray;
    if (!this->infer_instance_->inference(image, boxarray))
        return failure("Server error1");

    Json::Value boxarray_json(Json::arrayValue);
    for (auto &box : boxarray)
    {
        Json::Value item(Json::objectValue);
        item["left"] = box.left;
        item["top"] = box.top;
        item["right"] = box.right;
        item["bottom"] = box.bottom;
        item["confidence"] = box.confidence;
        item["class_label"] = box.class_label;
        item["class_name"] = cocolabels[box.class_label];
        boxarray_json.append(item);
    }
    return success(boxarray_json);
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

Json::Value LogicalController::detectBase64Image(const Json::Value &param)
{

    auto session = get_current_session();
    auto image_data = iLogger::base64_decode(session->request.body);
    if (image_data.empty())
        return failure("Image is required");

    auto image = cv::imdecode(image_data, 1);
    if (image_data.empty())
        return failure("Image is empty");

    Yolo::BoxArray boxarray;
    if (!this->infer_instance_->inference(image, boxarray))
        return failure("Server error1");

    Json::Value boxarray_json(Json::arrayValue);
    for (auto &box : boxarray)
    {
        Json::Value item(Json::objectValue);
        item["left"] = box.left;
        item["top"] = box.top;
        item["right"] = box.right;
        item["bottom"] = box.bottom;
        item["confidence"] = box.confidence;
        item["class_label"] = box.class_label;
        item["class_name"] = cocolabels[box.class_label];
        boxarray_json.append(item);
    }
    return success(boxarray_json);
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
/// ######################################
bool LogicalController::startup()
{

    infer_instance_.reset(new InferInstance());
    if (!infer_instance_->startup())
    {
        infer_instance_.reset();
    }
    return infer_instance_ != nullptr;
}

int test_http(int port = 8090)
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
        "7. http://%s                            访问web页面,vue开发的", 
        address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str(), address.c_str());

    INFO("按下Ctrl + C结束程序");
    return iLogger::while_loop();
}

int main(int argc, char **argv)
{
    InferYolo Ins(onnx_file, engine_file);

    cv::Mat image;
    const string videoStreamAddress = "rtsp://admin:admin123@192.168.0.213:554/cam/realmonitor?channel=1&subtype=0";
    cv::VideoCapture cap = cv::VideoCapture(videoStreamAddress);
    if (!cap.isOpened())
    {
        INFOE("Error opening video stream or file");
        return -1;
    }
    while (cap.read(image))
    {
        Yolo::BoxArray bbox;
        Ins.inference(image, bbox);
        for (auto &box : bbox)
        {
            cv::rectangle(image, cv::Point2d(box.left, box.top), cv::Point2d(box.right, box.bottom), cv::Scalar(color_list[box.class_label][0], color_list[box.class_label][1], color_list[box.class_label][2]), 3, 8, 0);
            auto caption = cv::format("%s %.3f", cocolabels[box.class_label], box.confidence);
            cv::putText(image, caption, cv::Point(box.left, box.top - 5), 0, 1, cv::Scalar(color_list[box.class_label][0], color_list[box.class_label][1], color_list[box.class_label][2]), 2, 16);
            // INFO("name: %s , Box:[%.2f, %.2f, %.2f, %.2f], confidence : %.3f", cocolabels[box.class_label], box.left, box.top, box.right, box.bottom, box.confidence);
        }
        cv::namedWindow("Output Window");
        imshow("Output Window", image);

        if (cv::waitKey(1) >= 0)
            break;
    }
    return 0;
}
