/*
 * @description:
 * @version:
 * @Author: zwy
 * @Date: 2022-11-12 20:23:29
 * @LastEditors: zwy
 * @LastEditTime: 2022-11-13 15:21:26
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
struct Box
{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
};

typedef std::vector<Box> Boxarray;

Boxarray NMS(Boxarray &bbox, float threshold)
{
    std::sort(bbox.begin(), bbox.end(), [](Box &a, Box &b)
              { return a.confidence > b.confidence; });
    Boxarray output;
    output.reserve(bbox.size());

    auto iou = [](Box &a, Box &b)
    {
        float cleft = std::max(a.left, b.left);
        float cright = std::max(a.right, b.right);
        float ctop = std::max(a.top, b.top);
        float cbottom = std::max(a.bottom, b.bottom);

        auto rectint = std::max(0.0f, cright - cleft) * std::max(0.0f, cbottom - ctop);
        auto unionrect = std::max(0.f, a.right - a.left) * std::max(0.f, a.bottom - a.top) + std::max(0.f, b.right - b.left) * std::max(0.f, b.bottom - b.top) - rectint;
        return rectint / unionrect;
    };

    std::vector<bool> remove_flags(bbox.size());
    for (int i = 0; i < bbox.size(); i++)
    {
        if (remove_flags[i])
            continue;

        auto &a = bbox[i];
        output.emplace_back(a);

        for (int j = i + 1; j < bbox.size(); j++)
        {
            if (remove_flags[j])
                continue;

            auto &b = bbox[j];
            if (b.class_label == a.class_label)
            {
                if (iou(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return output;
}

int main(int argc, char const *argv[])
{
    /* code */
    Box b1, b2, b3, b4, b5;
    b1.left = 10;
    b1.right = 30;
    b1.top = 15;
    b1.bottom = 45;
    b1.class_label = 1;
    b1.confidence = 0.94;

    b2.left = 20;
    b2.right = 30;
    b2.top = 45;
    b2.bottom = 75;
    b2.class_label = 2;
    b2.confidence = 0.51;

    b3.left = 12;
    b3.right = 34;
    b3.top = 13;
    b3.bottom = 35;
    b3.class_label = 1;
    b3.confidence = 0.78;

    b4.left = 13;
    b4.right = 52;
    b4.top = 25;
    b4.bottom = 55;
    b4.class_label = 2;
    b4.confidence = 0.88;

    b5.left = 123;
    b5.right = 152;
    b5.top = 222;
    b5.bottom = 265;
    b5.class_label = 1;
    b5.confidence = 0.68;

    std::vector<Box> bboxs = {b1, b2, b3, b4, b5};
    std::vector<Box> out = NMS(bboxs, 0.5);
    for (auto &box : out)
    {
       printf("name: %d , Box:[%.2f, %.2f, %.2f, %.2f], confidence : %.3f \n", box.class_label, box.left, box.top, box.right, box.bottom, box.confidence);
    }
    return 0;
}
