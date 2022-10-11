'''
Description: 
version: 
Author: zwy
Date: 2022-10-05 10:11:10
LastEditors: zwy
LastEditTime: 2022-10-05 13:15:13
'''
from cProfile import label
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.onnx
from unet import Unet
import torch.nn.functional as F
device = "cuda"
colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128,
                                                     0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128,
                                                        64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]

if __name__ == "__main__":
    unet = Unet()
    # dummy = torch.zeros(1, 3, 512, 512).cuda()
    # torch.onnx.export(
    #     unet.net, (dummy,), "../workspace/unet.onnx", input_names=["images"], output_names=["output"], opset_version=11,
    #     dynamic_axes={
    #         "images": {0: "batch"},
    #         "output": {0: "batch"}
    #     }
    # )
    # print("Done")

    model = unet.net
    model.eval().to(device)
    image = cv2.imread("img/street.jpg")
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image / 255.).astype(np.float32)
    image = image.transpose(2, 0, 1)[None]
    image = torch.from_numpy(image).to(device)

    with torch.no_grad():
        pr = model(image)
        # ptob = predict.softmax(dim=-1)
        pr = F.softmax(pr, dim=-1).cpu().numpy()
        pr = pr.argmax(axis=-1)
        seg_img = np.reshape(np.array(colors, np.uint8)[
            np.reshape(pr, [-1])], [512, 512, -1])
        # pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        print(seg_img.shape)
        pr = cv2.resize(seg_img, (512, 512), interpolation=cv2.INTER_LINEAR)

    # seg_img = np.reshape(np.array(colors, np.uint8)[
    #                      np.reshape(label_map, [-1])], [512, 512, -1])

    # image = Image.fromarray(np.uint8(pr))
    cv2.imwrite("image.jpg", pr)
