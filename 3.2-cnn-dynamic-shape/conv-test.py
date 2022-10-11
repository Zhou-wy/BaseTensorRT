'''
Descripttion: 
version: 
Author: zwy
Date: 2022-09-29 19:15:09
LastEditors: zwy
LastEditTime: 2022-09-29 19:51:05
'''
import torch
import torch.nn.functional as F

weight = torch.FloatTensor([
    [1.0, 2.0, 3.1],
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2],
]).view(1, 1, 3, 3)
bias = torch.FloatTensor([0.0]).view(1)
input = torch.FloatTensor([
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    [
        [-1, 1, 1],
        [1, 0, 1],
        [1, 1, -1],
    ]
]).view(2, 1, 3, 3)

print(F.conv2d(input, weight, bias, padding=1))

###
# tensor([[[[0.6000, 0.9000, 0.6000],
#           [5.7000, 7.0000, 3.6000],
#           [5.3000, 6.4000, 3.2000]]],


#         [[[0.2000, 0.5000, 0.4000],
#           [1.6000, 4.5000, 3.1000],
#           [2.2000, 4.2000, 2.0000]]]])
# ###
