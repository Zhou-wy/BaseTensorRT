/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-08 18:58:47
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-09 11:14:02
 */
#include <thread>
#include <mutex>

#include <queue>

#include "infer.hpp"

struct Job
{
    std::shared_ptr<std::promise<std ::string>> pro;
    std::string input;
};

class Infer : public BaseInfer
{
public:
    bool load_model(std::string &file)
    {
        //尽量保证资源在哪使用，就在哪里释放！
        //线程内返回值问题！！！
        std::promise<bool> pro;
        work_runing_ = true;
        work_thread_ = std::thread(&Infer::work, this, std::ref(file), std::ref(pro));
        return pro.get_future().get();
    }
    virtual std::shared_future<std::string> forward(std::string input) override
    {
        std::cout << "[info]: infer the image -> " << input << std::endl;
        Job job;
        job.pro.reset(new std::promise<std::string>());
        job.input = input;
        std::lock_guard<std::mutex> l(job_lock_);
        qjob_.push(job);

        //发送通知
        cv_.notify_one();
        return job.pro->get_future();
    }
    void destroy()
    {
        // context_.clear();
    }

    void work(std::string &file, std::promise<bool> &pro)
    {
        std::string context_ = file;
        if (context_.empty())
        {
            pro.set_value(false);
        }
        else
        {
            pro.set_value(true);
        }
        int max_batch_size = 5;
        std::vector<Job> jobs;
        int batch_id = 0;
        while (work_runing_)
        {
            // 等待通知
            std::unique_lock<std::mutex> l(job_lock_);
            cv_.wait(l, [&]()
                     {
                //为true, 推出等待
                //为false,继续等待
                return !qjob_.empty() || !work_runing_; });

            if (!work_runing_)
            {
                break;
            }

            while (jobs.size() < max_batch_size && !qjob_.empty())
            {
                jobs.emplace_back(qjob_.front());
                qjob_.pop();
            }

            // inference
            for (int i = 0; i < jobs.size(); i++)
            {
                char result[100];
                auto &job = jobs[i];
                sprintf(result, "%s : batch->%d[%ld]", job.input.c_str(), batch_id, jobs.size());

                job.pro->set_value(result);
            }
            batch_id++;
            jobs.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        printf("释放模型：%s \n", context_.c_str());
        printf("Work Done!");
    }
    virtual ~Infer(){
        work_runing_ = false;
        cv_.notify_one();
        if (work_thread_.joinable())
        {
            work_thread_.join();
        }
        
    }

private:
    std::atomic<bool> work_runing_{false};
    std::thread work_thread_;
    std::queue<Job> qjob_;
    std::mutex job_lock_;
    std::condition_variable cv_;
};

std::shared_ptr<BaseInfer> create_infer(std::string &file)
{
    std::shared_ptr<Infer> instance(new Infer());
    if (!instance->load_model(file))
    {
        instance.reset();
    }
    return instance;
}
