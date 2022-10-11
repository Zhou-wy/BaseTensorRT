'''
Description:
version: 
Author: zwy
Date: 2022-10-03 11:12:15
LastEditors: zwy
LastEditTime: 2022-10-03 15:46:27
'''
import torch
import torchvision

import cv2 as cv
import numpy as np

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained = True)

    def forward(self, x):
        feature = self.backbone(x)
        probability = torch.softmax(feature, dim = 1)
        return probability


# 对每个通道进行归一化有助于模型的训练
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
src = cv.imread("workspace/dog.jpg")
src = cv.resize(src, (224, 224))
src = src[..., ::-1]
src = src / 255.
src = (src - imagenet_mean) / imagenet_std
src = src.astype(np.float32)
src = src.transpose(2, 0, 1)
src = np.ascontiguousarray(src)
src = src[None, ...]
src = torch.from_numpy(src)
model = Classifier().eval()

with torch.no_grad():
    probability = model(src)

predict_class = probability.argmax(dim = 1).item()
confidence = probability[0, predict_class]

labels = open("workspace/labels.imagenet.txt").readlines()
labels = [item.strip() for item in labels]

print(f"Predict: {predict_class}, {confidence}, {labels[predict_class]}")

dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
        model, (dummy,), "bin/classifier.onnx",
        input_names = ["image"],
        output_names = ["prob"],
        dynamic_axes = {"image": {0: "batch"}, "prob": {0: "batch"}},
        opset_version = 11
)
