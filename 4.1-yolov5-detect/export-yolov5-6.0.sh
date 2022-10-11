###
 # @Description:
 # @version: 
 # @Author: zwy
 # @Date: 2022-10-04 10:13:11
 # @LastEditors: zwy
 # @LastEditTime: 2022-10-04 10:14:20
### 

cd yolov5
python export.py --weights=./yolov5s.pt --dynamic --include=onnx --opset=11

mv yolov5s.onnx ../workspace/yolov5s-raw.onnx