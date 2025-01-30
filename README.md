# ROS2 Object Detection Nodes
This repository implements a ROS2 system with a camera publisher node and a node that listens to the published image and runs the Faster-RCNN object detection algorithm. 

## Overview

### Image streamer

Images are captured from a USB-based camera and published to the 'camera/image' topic by [my_camera_pub.py](my_camera_pub.py). This node (CameraPublisher) publishes images at 30Hz.

### Object Detector

Images received from the 'camera/image' topic is used by the Faster RCNN Object Detection model to draw bounding boxes around the detected objects in [rcnn_node.py](rcnn_node.py). These predictions are also published to the 'camera/od_image' topic while simultaneosly being recorded and saved as an mp4 file.

### Listener

[listener.py](listener.py) implements a simple listner that also records the images received by it to an mp4 file.

## Pre-requisites

The Faster-RCNN model used in this repository was trained on the [Waymo Open Dataset](https://waymo.com/open/) and assumed you have the trained model file (frcnn_final_rr.pth). You may have to modify the labels_dict and ckpt_file variables in [rcnn_node.py](rcnn_node.py) as per your usecase.

## How to use this

- Clone this repository to the src dir of your [ros2 workspace](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html)

- Open a terminal session and run the camera publisher node (assuming you have already [sourced](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html#source-ros-2-environment) ros2)

```
python3 my_camera_pub.py
```

- Open another terminal session (again assuming you have sourced ros2) and run the object detector node

```
python3 rcnn_node.py
```

Use rviz to visualize the detections live. Or once you killed the above 2 programs, you can find the mp4 recording of the detections as output.mp4 