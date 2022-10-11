###
 # @Description:
 # @version: 
 # @Author: zwy
 # @Date: 2022-10-04 10:13:11
 # @LastEditors: zwy
 # @LastEditTime: 2022-10-04 10:34:42
### 

cd yolov5-change
python export.py --weights=./yolov5s.pt --dynamic --include=onnx --opset=11

mv yolov5s.onnx ../workspace/yolov5s.onnx