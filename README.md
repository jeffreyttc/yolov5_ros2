# yolov5_ros2
YOLOv5 with ROS2

Environment
* Ubuntu 20.04
* ROS2 Foxy
* OpenCV 4.2.0
* Python 3.8.10
* YOLOv5 - https://github.com/ultralytics/yolov5

Camera
* Webcam
* 
* Orbbec Astra
* ros2 run astra_camera astra_camera_node
* /image /depth
* ros2 run image_tools showimage --ros-args --remap image:=/image -p reliability:=best_effort

Execution
* ros2 run yolov5_ros2 
