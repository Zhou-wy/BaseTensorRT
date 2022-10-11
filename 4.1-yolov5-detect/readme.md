<!--
 * @Description: 
 * @version: 
 * @Author: zwy
 * @Date: 2022-10-04 16:17:23
 * @LastEditors: zwy
 * @LastEditTime: 2022-10-04 19:44:50
-->
# 知识点
1. yolov5的预处理部分，使用了仿射变换，请参照仿射变换原理
    - letterbox采用双线性插值对图像进行resize，并且使源图像和目标图像几何中心的对齐
        ![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/step1.png)
    - 使用仿射变换实现letterbox的理由是
        - 1. 便于操作，得到变换矩阵即可
        - 2. 便于逆操作，实则是逆矩阵映射即可
        - 3. 便于cuda加速，cuda版本的加速已经在cuda系列中提到了warpaffine实现
            - 该加速可以允许warpaffine、normalize、除以255、减均值除以标准差、变换RB通道等等在一个核中实现，性能最好
        ![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/step2.png)
2. 后处理部分，反算到图像坐标，实际上是乘以逆矩阵
    - 而由于逆矩阵实际上有效自由度是3，也就是d2i中只有3个数是不同的，其他都一样。也因此你看到的是d2i[0]、d2i[2]、d2i[5]在作用


# 运行步骤
1. 导出onnx模型
    - `bash export-yolov5-6.0.sh`
    - 脚本中会把模型文件移动到workspace/yolov5s.onnx下
2. 运行编译和推理
    - `make run -j64`

# 使用pytorch的yolov5进行导出
- 运行`bash detect-for-yolov5-6.0.sh`

# 修改过的地方：
```python
# line 55 forward function in yolov5/models/yolo.py 
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# modified into:

bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
ny = int(ny)
nx = int(nx)
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# line 70 in yolov5/models/yolo.py
#  z.append(y.view(bs, -1, self.no))
# modified into：
z.append(y.view(bs, self.na * ny * nx, self.no))

############# for yolov5-6.0 #####################
# line 65 in yolov5/models/yolo.py
# if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
#    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
# modified into:
if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

# disconnect for pytorch trace
anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(1, -1, 1, 1, 2)

# line 70 in yolov5/models/yolo.py
# y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh

# line 73 in yolov5/models/yolo.py
# wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
wh = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
############# for yolov5-6.0 #####################

# line 77 in yolov5/models/yolo.py
# return x if self.training else (torch.cat(z, 1), x)
# modified into:
return x if self.training else torch.cat(z, 1)

# line 52 in yolov5/export.py
# torch.onnx.export(dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)  修改为
# modified into:
torch.onnx.export(dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                'output': {0: 'batch'}  # shape(1,25200,85) 
```

## NMS 
**前提：目标边界框列表及其对应的置信度得分列表，设定阈值，阈值用来删除重叠较大的边界框。**
**IoU：intersection-over-union，即两个边界框的交集部分除以它们的并集。**

非极大值抑制的流程如下：

- 根据置信度得分进行排序
- 选择置信度最高的比边界框添加到最终输出列表中，将其从边界框列表中删除
- 计算所有边界框的面积
- 计算置信度最高的边界框与其它候选框的IoU。
- 删除IoU大于阈值的边界框
- 重复上述过程，直至边界框列表为空。

```c++
std::vector<std::vector<float>> NMS(std::vector<std::vector<float>> &bboxes, float nms_threshold)
{
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float> &a, std::vector<float> &b)
              { return a[5] > b[5]; });
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> result_box;
    result_box.reserve(bboxes.size());

    auto iou = [](const std::vector<float> &a, const std::vector<float> &b)
    {
        float cross_left = std::max(a[0], b[0]);
        float cross_top = std::max(a[1], b[1]);
        float cross_right = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if (cross_area == 0 || union_area == 0)
            return 0.0f;
        return cross_area / union_area;
    };
    for (int i = 0; i < bboxes.size(); i++)
    {
        if (remove_flags[i])
            continue;
        auto &ibox = bboxes[i];
        result_box.emplace_back(ibox);
        for (int j = i + 1; j < bboxes.size(); j++)
        {
            if (remove_flags[i])
                continue;
            auto &jbox = bboxes[j];
            if (ibox[4] == jbox[4])
                if (iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
        }
    }
    return result_box;
}
```
