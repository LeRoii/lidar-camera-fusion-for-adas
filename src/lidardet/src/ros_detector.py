# ros package
import rospy
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point, Point32, Pose
# from derived_object_msgs.msg import ObjectArray, Object

from std_msgs.msg import Header


import numpy as np
import os
import sys
import torch
import time
import glob

from pathlib import Path
from pcdet.datasets import DatasetTemplate
from pyquaternion import Quaternion
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from objectbase import objectbase, makeobjects
from tracker import tracker

import rosbag

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path/ f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfil(self.sample_file_list[index], dtype=np.float32).reshape(-1,4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)


def remove_low_score_ck(box_dict, class_scores=None):
    pred_scores = box_dict['pred_scores'].detach().cpu().numpy()
    pred_boxes = box_dict['pred_boxes'].detach().cpu().numpy()
    pred_labels = box_dict['pred_labels'].detach().cpu().numpy()

    if class_scores is None:
        return box_dict

    keep_indices = []
    for i in range(pred_scores.shape[0]):
        if pred_scores[i] >= class_scores[pred_labels[i]-1]:
            keep_indices.append(i)
    for key in box_dict:
        box_dict[key] = box_dict[key][keep_indices]
    return box_dict


def transform_to_original(boxes_lidar):
    # boxes_lidar:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    transformed_boxes_lidar = boxes_lidar.copy()
    transformed_boxes_lidar[:, 0] = -boxes_lidar[:, 1]
    transformed_boxes_lidar[:, 1] = boxes_lidar[:, 0]
    transformed_boxes_lidar[:, 2] = boxes_lidar[:, 2] - 2.0

    transformed_boxes_lidar[:, 3] = boxes_lidar[:, 4]
    transformed_boxes_lidar[:, 4] = boxes_lidar[:, 3]

    return transformed_boxes_lidar


class Precessor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None

    def initialize(self):
        self.read_config()

    def read_config(self):
        print(self.config_path)
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_datasets  = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path('/home/iairiv/lidar_network/KITTI/2011_09_26/2011_09_26_drive_0015_sync/velodyne_points/data/0000000000.bin'),
            ext='.bin'
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_datasets)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def run(self, points):
        t1 = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 4 # kitti model
        # num_features = 5    # nuscene model
        if num_features == 5 and points.shape[1] == 4:
            self.points = np.zeros((points.shape[0], 5))
            self.points[:, 0:4] = points
        else:
            self.points = points.reshape([-1, num_features])

        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_datasets.prepare_data(data_dict=input_dict)
        data_dict = self.demo_datasets.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        pred_dicts, _ = self.net.forward(data_dict)
        torch.cuda.synchronize()

        t2 = time.time()
        print(f"net inference cost time: {t2 - t1}")

        # pred = remove_low_score_nu(pred_dicts[0], 0.45)

        # 'vehicle', 'pedestrian', 'bicycle'
        # class_scores = [0.5, 0.30, 0.30]
        # class_scores = [0.5, 0.5, 0.3, 0.3]
        class_scores = [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3]
        # 'car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        # class_scores = [0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]       # nuscene

        pred = remove_low_score_ck(pred_dicts[0], class_scores)

        boxes_lidar = pred['pred_boxes'].detach().cpu().numpy()
        boxes_lidar = transform_to_original(boxes_lidar)
        scores = pred['pred_scores'].detach().cpu().numpy()
        types = pred['pred_labels'].detach().cpu().numpy()
        #print(f" pred boxes: { boxes_lidar }")
        print(f" pred labels: {types}")
        print(f" pred scores: {scores}")
        #print(pred_dicts)

        return scores, boxes_lidar, types


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype) # kitti model
    #points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    #points = np.zeros(cloud_array.shape, dtype=dtype)

    # our received lidar point cloud is :      front->y     right->x
    # net need is :                            front->x     left->y
    points[..., 0] = cloud_array['y']
    points[..., 1] = -cloud_array['x']
    points[..., 2] = cloud_array['z'] + 2.0  # robosense

    # points[..., 0] = cloud_array['x']
    # points[..., 1] = cloud_array['y']
    # points[..., 2] = cloud_array['z'] + 0.2  # kitti

    # points[..., 2] = cloud_array['z'] + 0.2  # nuscene
    return points


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    msg = PointCloud2()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        #PointField('i', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.array(points_sum, np.float32).tobytes()
    return msg


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    #print(is_numpy)
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    a = boxes3d[:, None, 3:6]
    b = template[None, :, :]
    c = boxes3d[:, None, 3:6].repeat(1, 8, 1)
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


lidartracker = tracker()
def lidar_callback(msg):
    print('lidar_callback')
    t1 = time.time()
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    frame_id = msg.header.frame_id
    np_p = get_xyz_points(msg_cloud, True)
    # print(msg.header.stamp.to_sec())
    # print(rospy.Time.now().to_sec())

    # print(np_p.size)
    scores, dt_box_lidar, types = proc_1.run(np_p)

    np_p[:, 2] -= 2.0
    bbox_corners3d = boxes_to_corners_3d(dt_box_lidar)
    corners = bbox_corners3d.reshape((bbox_corners3d.shape[0],-1))

    netret = np.column_stack((dt_box_lidar, scores, types, corners))
    objs = makeobjects(netret)
    lidartracker.update(objs)

    empty_markers = MarkerArray()
    clear_marker = Marker()
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.header.frame_id = frame_id
    clear_marker.ns = "tracked objects"
    clear_marker.id = 0
    clear_marker.action = clear_marker.DELETEALL
    clear_marker.lifetime = rospy.Duration()
    empty_markers.markers.append(clear_marker)
    pub_bbox_array.publish(empty_markers)

    bbox_arry = MarkerArray()
    for i in range(scores.size):
        point_list = []
        bbox = Marker()
        bbox.type = Marker.LINE_LIST
        bbox.ns = "detected objects"
        bbox.id = i+100
        bbox.pose = Pose(orientation = Quaternion(x=0,y=0,z=0,w=1))
        bbox.lifetime = rospy.Duration()
        box = bbox_corners3d[i]
        for j in range(24):
            p = Point()
            point_list.append(p)
        
        cornerStartPt = [[0,1,2,3],[4,5,6,7],[0,1,2,3]]
        cornerEndPt = [[1,2,3,0],[5,6,7,4],[4,5,6,7]]
        for i in range(3):
            for j in range(4):
                markerStartPt = (i*4+j)*2
                markerEndPt = markerStartPt+1
                point_list[markerStartPt].x = box[cornerStartPt[i][j],0]
                point_list[markerStartPt].y = box[cornerStartPt[i][j],1]
                point_list[markerStartPt].z = box[cornerStartPt[i][j],2]
                point_list[markerEndPt].x = box[cornerEndPt[i][j],0]
                point_list[markerEndPt].y = box[cornerEndPt[i][j],1]
                point_list[markerEndPt].z = box[cornerEndPt[i][j],2]

        for j in range(24):
            bbox.points.append(point_list[j])
        bbox.scale.x = 0.1
        bbox.color.a = 1.0
        bbox.color.r = 0.0
        bbox.color.g = 1.0
        bbox.color.b = 0.0
        #bbox.header.stamp = rospy.Time.now()
        bbox.header.stamp = msg.header.stamp
        bbox.header.frame_id = frame_id
        bbox_arry.markers.append(bbox)

    bboxid = 0
    for tracked in lidartracker.trackerList:
        if tracked.age < 5:
            continue
        else:
            point_list = []
            bbox = Marker()
            bbox.type = Marker.LINE_LIST
            bbox.ns = "tracked objects"
            bbox.id = bboxid
            bboxid += 1
            bbox.pose = Pose(orientation = Quaternion(x=0,y=0,z=0,w=1))
            for j in range(24):
                p = Point()
                point_list.append(p)
            
            cornerStartPt = [[0,1,2,3],[4,5,6,7],[0,1,2,3]]
            cornerEndPt = [[1,2,3,0],[5,6,7,4],[4,5,6,7]]
            for i in range(3):
                for j in range(4):
                    markerStartPt = (i*4+j)*2
                    markerEndPt = markerStartPt+1
                    point_list[markerStartPt].x = tracked.bbox3d.corners[cornerStartPt[i][j]].x
                    point_list[markerStartPt].y = tracked.bbox3d.corners[cornerStartPt[i][j]].y
                    point_list[markerStartPt].z = tracked.bbox3d.corners[cornerStartPt[i][j]].z
                    point_list[markerEndPt].x = tracked.bbox3d.corners[cornerEndPt[i][j]].x
                    point_list[markerEndPt].y = tracked.bbox3d.corners[cornerEndPt[i][j]].y
                    point_list[markerEndPt].z = tracked.bbox3d.corners[cornerEndPt[i][j]].z

            for j in range(24):
                bbox.points.append(point_list[j])
            bbox.scale.x = 0.1
            bbox.color.a = 1.0
            bbox.color.r = 1.0
            bbox.color.g = 0.0
            bbox.color.b = 1.0
            #bbox.header.stamp = rospy.Time.now()
            bbox.header.stamp = msg.header.stamp
            bbox.header.frame_id = frame_id
            bbox_arry.markers.append(bbox)

            # add text
            text_show = Marker()
            text_show.type = Marker.TEXT_VIEW_FACING
            text_show.ns = "tracked objects"
            text_show.header.stamp = msg.header.stamp
            text_show.header.frame_id = frame_id
            text_show.id = bbox.id + len(lidartracker.trackerList)
            text_show.pose = Pose(
                position=Point(float(tracked.bbox3d.centerpoint.x),
                               float(tracked.bbox3d.centerpoint.y),
                               float(tracked.bbox3d.centerpoint.z+2.0)), orientation=Quaternion(x=0, y=0, z=0, w=1)
            )
            distance_obj = np.sqrt(tracked.bbox3d.centerpoint.x*tracked.bbox3d.centerpoint.x + tracked.bbox3d.centerpoint.y*tracked.bbox3d.centerpoint.y)
            text_show.text = str(proc_1.net.class_names[int(tracked.objtype)-1]) + ' ' + str(tracked.id) + ' ' \
                + str(round(tracked.confidence, 2)) + '\n' + str(round(distance_obj, 2)) + ' ' + str(tracked.age) + ' ' + str(tracked.lostCnt)
            text_show.action = Marker.ADD
            text_show.color.a = 1.0
            text_show.color.r = 1.0
            text_show.color.g = 1.0
            text_show.color.b = 0.0
            text_show.scale.z = 1.5
            bbox_arry.markers.append(text_show)

    pub_bbox_array.publish(bbox_arry)
    pass

    # empty_markers = MarkerArray()
    # clear_marker = Marker()
    # clear_marker.header.stamp = rospy.Time.now()
    # clear_marker.header.frame_id = frame_id
    # clear_marker.ns = "objects"
    # clear_marker.id = 0
    # clear_marker.action = clear_marker.DELETEALL
    # clear_marker.lifetime = rospy.Duration()
    # empty_markers.markers.append(clear_marker)
    # pub_bbox_array.publish(empty_markers)

    # bbox_arry = MarkerArray()
    # if scores.size != 0:
    #     for i in range(scores.size):
    #         point_list = []
    #         bbox = Marker()
    #         bbox.type = Marker.LINE_LIST
    #         bbox.ns = "objects"
    #         bbox.id = i
    #         box = bbox_corners3d[i]
    #         q = yaw2quaternion(float(dt_box_lidar[i][6]))
    #         for j in range(24):
    #             p = Point()
    #             point_list.append(p)

    #         point_list[0].x = float(box[0, 0])
    #         point_list[0].y = float(box[0, 1])
    #         point_list[0].z = float(box[0, 2])
    #         point_list[1].x = float(box[1, 0])
    #         point_list[1].y = float(box[1, 1])
    #         point_list[1].z = float(box[1, 2])

    #         point_list[2].x = float(box[1, 0])
    #         point_list[2].y = float(box[1, 1])
    #         point_list[2].z = float(box[1, 2])
    #         point_list[3].x = float(box[2, 0])
    #         point_list[3].y = float(box[2, 1])
    #         point_list[3].z = float(box[2, 2])

    #         point_list[4].x = float(box[2, 0])
    #         point_list[4].y = float(box[2, 1])
    #         point_list[4].z = float(box[2, 2])
    #         point_list[5].x = float(box[3, 0])
    #         point_list[5].y = float(box[3, 1])
    #         point_list[5].z = float(box[3, 2])

    #         point_list[6].x = float(box[3, 0])
    #         point_list[6].y = float(box[3, 1])
    #         point_list[6].z = float(box[3, 2])
    #         point_list[7].x = float(box[0, 0])
    #         point_list[7].y = float(box[0, 1])
    #         point_list[7].z = float(box[0, 2])

    #         point_list[8].x = float(box[4, 0])
    #         point_list[8].y = float(box[4, 1])
    #         point_list[8].z = float(box[4, 2])
    #         point_list[9].x = float(box[5, 0])
    #         point_list[9].y = float(box[5, 1])
    #         point_list[9].z = float(box[5, 2])

    #         point_list[10].x = float(box[5, 0])
    #         point_list[10].y = float(box[5, 1])
    #         point_list[10].z = float(box[5, 2])
    #         point_list[11].x = float(box[6, 0])
    #         point_list[11].y = float(box[6, 1])
    #         point_list[11].z = float(box[6, 2])

    #         point_list[12].x = float(box[6, 0])
    #         point_list[12].y = float(box[6, 1])
    #         point_list[12].z = float(box[6, 2])
    #         point_list[13].x = float(box[7, 0])
    #         point_list[13].y = float(box[7, 1])
    #         point_list[13].z = float(box[7, 2])

    #         point_list[14].x = float(box[7, 0])
    #         point_list[14].y = float(box[7, 1])
    #         point_list[14].z = float(box[7, 2])
    #         point_list[15].x = float(box[4, 0])
    #         point_list[15].y = float(box[4, 1])
    #         point_list[15].z = float(box[4, 2])

    #         point_list[16].x = float(box[0, 0])
    #         point_list[16].y = float(box[0, 1])
    #         point_list[16].z = float(box[0, 2])
    #         point_list[17].x = float(box[4, 0])
    #         point_list[17].y = float(box[4, 1])
    #         point_list[17].z = float(box[4, 2])

    #         point_list[18].x = float(box[1, 0])
    #         point_list[18].y = float(box[1, 1])
    #         point_list[18].z = float(box[1, 2])
    #         point_list[19].x = float(box[5, 0])
    #         point_list[19].y = float(box[5, 1])
    #         point_list[19].z = float(box[5, 2])

    #         point_list[20].x = float(box[2, 0])
    #         point_list[20].y = float(box[2, 1])
    #         point_list[20].z = float(box[2, 2])
    #         point_list[21].x = float(box[6, 0])
    #         point_list[21].y = float(box[6, 1])
    #         point_list[21].z = float(box[6, 2])

    #         point_list[22].x = float(box[3, 0])
    #         point_list[22].y = float(box[3, 1])
    #         point_list[22].z = float(box[3, 2])
    #         point_list[23].x = float(box[7, 0])
    #         point_list[23].y = float(box[7, 1])
    #         point_list[23].z = float(box[7, 2])

    #         for j in range(24):
    #             bbox.points.append(point_list[j])
    #         bbox.scale.x = 0.1
    #         bbox.color.a = 1.0
    #         bbox.color.r = 1.0
    #         bbox.color.g = 0.0
    #         bbox.color.b = 0.0
    #         #bbox.header.stamp = rospy.Time.now()
    #         bbox.header.stamp = msg.header.stamp
    #         bbox.header.frame_id = frame_id
    #         bbox_arry.markers.append(bbox)

            # add text
            # text_show = Marker()
            # text_show.type = Marker.TEXT_VIEW_FACING
            # text_show.ns = "objects"
            # text_show.header.stamp = msg.header.stamp
            # text_show.header.frame_id = frame_id
            # text_show.id = i + scores.size
            # text_show.pose = Pose(
            #     position=Point(float(dt_box_lidar[i][0]),
            #                    float(dt_box_lidar[i][1]),
            #                    float(dt_box_lidar[i][2])+2.0)
            # )
            # distance_obj = np.sqrt(dt_box_lidar[i][0]*dt_box_lidar[i][0] + dt_box_lidar[i][1]*dt_box_lidar[i][1])
            # text_show.text = str(proc_1.net.class_names[types[i]-1]) + ' ' + str(round(scores[i], 2)) + '\n' + str(round(distance_obj, 2))
            # text_show.action = Marker.ADD
            # text_show.color.a = 1.0
            # text_show.color.r = 1.0
            # text_show.color.g = 1.0
            # text_show.color.b = 0.0
            # text_show.scale.z = 1.5
            # bbox_arry.markers.append(text_show)

    # print("total callback time: ", time.time() - t1)

    # pub_bbox_array.publish(bbox_arry)
    # cloud_array = xyz_array_to_pointcloud2(np_p[:, :3], rospy.Time.now(), frame_id)
    # pub_point2_.publish(cloud_array)

    # ## publish to fusion
    # msg_lidar_objects = ObjectArray()
    # msg_lidar_objects.header.stamp = msg.header.stamp
    # msg_lidar_objects.header.frame_id = frame_id

    # # get points in each box
    # t3 = time.time()
    # num_obj = dt_box_lidar.shape[0]
    # point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
    #     torch.from_numpy(np_p[:, 0:3]), torch.from_numpy(dt_box_lidar)
    # ).numpy()  # (nboxes, npoints)

    # t4 = time.time()
    # print(f"get points in each box cost time: {t4 - t3}")
    # print('')

    # #clustered_points = []       #for test
    # if scores.size != 0:
    #     for i in range(scores.size):
    #         lidar_object = Object()
    #         lidar_object.header.stamp = msg.header.stamp
    #         lidar_object.header.frame_id = frame_id
            
    #         lidar_object.id = i
    #         lidar_object.pose.position.x = float(dt_box_lidar[i][0])
    #         lidar_object.pose.position.y = float(dt_box_lidar[i][1])
    #         lidar_object.pose.position.z = float(dt_box_lidar[i][2])
    #         lidar_object.pose.orientation = yaw2quaternion(float(dt_box_lidar[i][6]))
    #         lidar_object.shape.dimensions.append(float(dt_box_lidar[i][3]))
    #         lidar_object.shape.dimensions.append(float(dt_box_lidar[i][4]))
    #         lidar_object.shape.dimensions.append(float(dt_box_lidar[i][5]))
    #         type_Plan = -1
    #         if types[i] == 1:
    #             type_Plan = 1
    #         elif types[i] == 2:
    #             type_Plan = 1
    #         elif types[i] == 3:
    #             type_Plan = 2
    #         elif types[i] == 4:
    #             type_Plan = 3
    #         elif types[i] == 5:
    #             type_Plan = 5
    #         elif types[i] == 6:
    #             type_Plan = 4
    #         elif types[i] == 7:
    #             type_Plan = 0
    #         elif types[i] == 8:
    #             type_Plan = 6
                

    #         lidar_object.classification = type_Plan

    #         gt_points = np_p[point_indices[i] > 0]
    #         for pp in range(gt_points.shape[0]):
    #             temp_point = Point32()
    #             temp_point.x = gt_points[pp][0]
    #             temp_point.y = gt_points[pp][1]
    #             temp_point.z = gt_points[pp][2]
    #             lidar_object.polygon.points.append(temp_point)
    #             #clustered_points.append(gt_points[pp,:].tolist())

    #         msg_lidar_objects.objects.append(lidar_object)
    
    # # clustered_points = np.array(clustered_points)
    # # cloud_array = xyz_array_to_pointcloud2(clustered_points[:, :3], rospy.Time.now(), frame_id)
    # # pub_point2_.publish(cloud_array)

    # pub_object_array.publish(msg_lidar_objects)


if __name__ == "__main__":
    global proc
    ## config and model path
    #################################   KITTI   ##############################################################
    # config_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/kitti_models_ck/pointpillar/baseline_20220131/pointpillar.yaml'
    # modle_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/kitti_models_ck/pointpillar/baseline_20220131/ckpt/checkpoint_epoch_80.pth'

    # config_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/kitti_models_ck/centernet_multihead/0302_sighead_gassiu/centernet_multihead.yaml'
    # modle_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/kitti_models_ck/centernet_multihead/0302_sighead_gassiu/ckpt/checkpoint_epoch_80.pth'

    # config_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/kitti_models_ck/centernet_multihead_twostage/twostage_auxcorner/centernet_multihead_twostage.yaml'
    # modle_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/kitti_models_ck/centernet_multihead_twostage/twostage_auxcorner/ckpt/checkpoint_epoch_80.pth'
    
    #################################   Robosense   ##############################################################
    # config_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/robosense_models/robosense_pointpillar/test_first/robosense_pointpillar.yaml'
    # modle_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/robosense_models/robosense_pointpillar/test_first/ckpt/checkpoint_epoch_80.pth'

    config_path = '/space/code/sqadas/rosws/src/lidardet/model/output/robosense_models/robosense_centernet_multi/0119_stride4/robosense_centernet_multi.yaml'
    modle_path = '/space/code/sqadas/rosws/src/lidardet/model/output/robosense_models/robosense_centernet_multi/0119_stride4/ckpt/checkpoint_epoch_80.pth'

    # config_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/robosense_models/robosense_centernet_v2_2_24/first_0224/robosense_centernet_v2_2_24.yaml'
    # modle_path = '/home/iairiv/lidar_network/OpenLidarPerceptron/output/robosense_models/robosense_centernet_v2_2_24/first_0224/ckpt/checkpoint_epoch_40.pth'

    #################################   nuscenes   ##############################################################
    # config_path = '/home/syang/Data/nuscenes/output/nuscenes_models/cbgs_pp_multihead/first/cbgs_pp_multihead.yaml'
    # modle_path = '/home/syang/Data/nuscenes/output/nuscenes_models/cbgs_pp_multihead/first/ckpt/checkpoint_epoch_20.pth'

    # config_path = '/home/syang/Data/nuscenes/output/nuscenes_models/cbgs_centernet_multihead/first_0228/cbgs_centernet_multihead.yaml'
    # modle_path = '/home/syang/Data/nuscenes/output/nuscenes_models/cbgs_centernet_multihead/first_0228/ckpt/checkpoint_epoch_15.pth'

    proc_1 = Precessor_ROS(config_path, modle_path)

    proc_1.initialize()

    # points = np.load('/home/syang/lidar_network/128/128.npy')
    # proc_1.run(points)
    # proc_1.run(points)
    # print()

    rospy.init_node('net_lidar_ros_node')
    # sub_lidar_topic = [
    #     "/velodyne_points",
    #     "/rslidar_points",
    #     "/kitti/velo/pointcloud",
    #     "/calibrated_cloud",
    #     "/segmenter/points_nonground"
    # ]
    # sub_ = rospy.Subscriber(sub_lidar_topic[1], PointCloud2, lidar_callback, queue_size=1, buff_size=2**24)

    pub_bbox_array = rospy.Publisher('lidar_net_results', MarkerArray, queue_size=1)
    pub_point2_ = rospy.Publisher('lidar_points', PointCloud2, queue_size=1)
    # pub_object_array = rospy.Publisher('lidar_DL_objects', ObjectArray, queue_size=1)

    # print("lidar net ros start!")
    # # rate = rospy.Rate(10)
    # rospy.spin()
    # # rate.sleep()

    bagfile = '/space/data/sq/rslidar32.bag'
    bag = rosbag.Bag(bagfile,'r')
    print(bag.get_type_and_topic_info())
    pub_pt = rospy.Publisher('/rslidar_points', PointCloud2, queue_size=1)
    bag_data = bag.read_messages('/rslidar_points')
    for topic, msg, t in bag_data:
        pub_pt.publish(msg)
        lidar_callback(msg)
        time.sleep(0.05)

