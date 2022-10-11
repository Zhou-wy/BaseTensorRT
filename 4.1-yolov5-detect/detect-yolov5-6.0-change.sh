
###
 # @Description: 
 # @version: 
 # @Author: zwy
 # @Date: 2022-10-04 11:02:27
 # @LastEditors: zwy
 # @LastEditTime: 2022-10-04 16:35:37
### 

cd yolov5-change

python detect.py --weights=./yolov5s.pt --source=../workspace/test.jpg --iou-thres=0.5 --conf-thres=0.25 --project=../workspace/ --line-thickness=10

mv ../workspace/exp/test.jpg ../workspace/test-pytorch.jpg
rm -rf ../workspace/exp