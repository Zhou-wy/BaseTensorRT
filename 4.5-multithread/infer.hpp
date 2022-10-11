/*
 * @Description:
 * @version:
 * @Author: zwy
 * @Date: 2022-10-08 18:58:55
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-09 11:04:43
 */
#ifndef __INFER_HPP
#define __INFER_HPP
#include <string>
#include <iostream>
#include <memory>
#include <future>
class BaseInfer
{
public:
    virtual std::shared_future<std::string> forward(std::string input) = 0;
};

std::shared_ptr<BaseInfer> create_infer(std::string &file);
#endif //__INFER_HPP