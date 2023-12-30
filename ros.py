#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import PointCloud2, PointField

import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import time
import numpy as np
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import scipy.linalg as linalg
import time
import sys
import os

sys.path.append("xxx/src/mmdet")

from argparse import ArgumentParser
import math
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.core.points import get_points_type
from std_msgs.msg import Float64MultiArray

class Pointpillars_ROS:

    def lidar_callback(self, msg):

        # build the model from a config file and a checkpoint file
        # 填写自己的实际路径
        checkpoint_file = 'xxx/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        config_file = 'xxx/tools/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        
        # 在回调函数中
        #######The filed_names depend on your dataset#######
        points = np.array(pc2.read_points_list(msg, field_names = ("x", "y", "z","intensity"), skip_nans=True))
        ######## 连接激光雷达进行实时检测时可以将'intensity'这一项置为0，不然有可能检测错误 ########
        points[:, [3]] = 0
        points_class = get_points_type('LIDAR')
        ###### Here the points_dim is 4 ######
        points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        result, data = inference_detector(model, points_mmdet3d)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        print(f'single frame: {elapsed*1000:.1f} ms')
        scores = result[0]['scores_3d'].numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = result[0]['boxes_3d'][mask].tensor.cpu().numpy()
        label = result[0]['labels_3d'][mask].numpy()
        arr_bbox = BoundingBoxArray()
        for i in range(boxes_lidar.shape[0]):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])  # + float(boxes_lidar[5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]
            # print(label[i])
            arr_bbox.boxes.append(bbox)
        arr_bbox.header.frame_id = msg.header.frame_id
        if len(arr_bbox.boxes) is not 0:
            pub_bbox.publish(arr_bbox)
            self.publish_test(points, msg.header.frame_id)

    def publish_test(self, cloud, frame_id):
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = frame_id
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]  # ,PointField('label', 16, PointField.FLOAT32, 1)
        # creat_cloud不像read，他必须要有fields,而且field定义有两种。一个是上面的，一个是下面的fields=_make_point_field(4)
        msg_segment = pc2.create_cloud(header=header, fields=fields, points=cloud)

        pub_velo.publish(msg_segment)
        # pub_image.publish(image_msg)

if __name__ == '__main__':
    global sec
    sec = Pointpillars_ROS()

    rospy.init_node('pointpillars_ros_node', anonymous=True)

    # subscriber

    sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, sec.lidar_callback, queue_size=1,
                                buff_size=2 ** 12)

    # publisher
    pub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
    
    pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
