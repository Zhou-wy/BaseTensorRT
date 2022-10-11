/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-07 15:09:10
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-09 11:06:18
 */
#include <iostream>
#include <thread>
#include <chrono>
#include "infer.hpp"
/*
void worker(int a, std::string &str)
{
    std::cout << "hello world! " << a << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "work done" << std::endl;
    str = "output";
}
class Infer
{
public:
    Infer()
    {
        work_thread_ = std::thread(&Infer::infer_worker, this);
    }

private:
    std::thread work_thread_;
    void infer_worker()
    {

    }
};
*/
/*
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <memory>

std::mutex lock_;
std::condition_variable cv_;
int limit_ = 5;
struct Job
{
    std::shared_ptr<std::promise<std::string>> prob;
    std::string input;
};
std::queue<Job> qjob_;

void video_capture()
{
    int pic_id = 0;
    while (true)
    {
        Job job;
        {
            std::unique_lock<std::mutex> l(lock_);
            char name[100];
            sprintf(name, "PIC-%d", pic_id++);
            printf("生产了一个新图片 %s, qjob_.size = %ld\n", name, qjob_.size());
            cv_.wait(l, [&]()
                     { return qjob_.size() < limit_; });

            job.input = name;
            job.prob.reset(new std::promise<std::string>());
            qjob_.push(job);
        }
        auto res = job.prob->get_future().get();
        printf("Job %s -> %s \n", job.input.c_str(), res.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void Infer_worker()
{
    while (true)
    {
        if (!qjob_.empty())
        {
            {
                std::lock_guard<std::mutex> l(lock_);
                auto p_job = qjob_.front();
                qjob_.pop();
                cv_.notify_one();
                printf("消费量一张图片 %s\n", p_job.input.c_str());

                auto new_pic = p_job.input + " ---infer";
                p_job.prob->set_value(new_pic);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        std::this_thread::yield();
    }
}
*/
int main(int argc, char const *argv[])
{
    /*
    std::thread t0(video_capture);
    std::thread t1(Infer_worker);
    if (t0.joinable())
        t0.join();
    if (t1.joinable())
        t1.join();
    */
    std::string model = "demo.onnx";
    auto infer = create_infer(model);
    auto f1 = infer->forward("imag1.jpg");
    auto f2 = infer->forward("imag2.jpg");
    auto f3 = infer->forward("imag3.jpg");

    printf("%s\n", f1.get().c_str());
    printf("%s\n", f2.get().c_str());
    printf("%s\n", f3.get().c_str());

    return 0;
}
