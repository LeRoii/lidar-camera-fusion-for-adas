# ros package
import rospy
import ros_numpy
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point, Point32, Pose

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
from fusion import fusion

import rosbag
from sensor_msgs.msg import Image
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
from predict import Predictor
import tf
import tf2_ros
import geometry_msgs.msg
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from objmsg.msg import obj,objArray
import threading

from sensor_msgs.msg import NavSatFix
import math

lastimgmsgtime = 0
lastlidarmsgtime = 0

ellipse_a  = 6378137
ellipse_e  = 0.081819190842622

basePoint = [34.2569999, 108.6511768, 392.931]
mat = np.zeros((4,3), dtype='float')

def generateMat(lat,lon,height):
    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi
    sinLati = math.sin(lat)
    cosLati = math.cos(lat)
    sinLong = math.sin(lon) 
    cosLong = math.cos(lon)

    N = ellipse_a / math.sqrt((1-ellipse_e*ellipse_e*sinLati*sinLati))

    mat[1][0] = -sinLong
    mat[1][1] = cosLong
    mat[1][2] = 0
    mat[0][0] = -sinLati*cosLong
    mat[0][1] = -sinLati*sinLong
    mat[0][2] = cosLati
    mat[2][0] = cosLati*cosLong
    mat[2][1] = cosLati*sinLong
    mat[2][2] = sinLati

    mat[3][0] = (N + height)*cosLati*cosLong
    mat[3][1] = (N + height)*cosLati*sinLong
    mat[3][2] = (N*(1 - ellipse_e*ellipse_e) + height)*sinLati

    return mat

def gps2ENU(lat,lon,height):
    x0 = mat[3][0]
    y0 = mat[3][1]
    z0 = mat[3][2]

    lat = lat / 180 * math.pi
    lon = lon / 180 * math.pi
    sinLati = math.sin(lat)
    cosLati = math.cos(lat)
    sinLong = math.sin(lon) 
    cosLong = math.cos(lon)

    N = ellipse_a / math.sqrt((1-ellipse_e*ellipse_e*sinLati*sinLati))
    x1 = (N + height)*cosLati*cosLong
    y1 = (N + height)*cosLati*sinLong
    z1 = (N*(1 - ellipse_e*ellipse_e) + height)*sinLati
    dx = x1 - x0 
    dy = y1 - y0 
    dz = z1 - z0
    outputy = mat[0][0] * dx + mat[0][1] * dy + mat[0][2] * dz
    outputx = mat[1][0] * dx + mat[1][1] * dy
    outputz = mat[2][0] * dx + mat[2][1] * dy + mat[2][2] * dz

    return outputx,outputy,outputz

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
        # print(f" pred labels: {types}")
        # print(f" pred scores: {scores}")
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

    # points[..., 0] = cloud_array['x']
    # points[..., 1] = cloud_array['y']
    # points[..., 2] = cloud_array['z'] + 1.0  # sq

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
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

lidartracker = tracker()
# predictor = Predictor(120)
fusionnode = fusion()

def lidar_callback(msg):
    global lastlidarmsgtime
    print('lidar_callback:::msg time:', msg.header.stamp.to_sec())
    print('lidar time diff:', msg.header.stamp.to_sec() - lastlidarmsgtime)
    lastlidarmsgtime = msg.header.stamp.to_sec()
    
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    frame_id = msg.header.frame_id
    np_p = get_xyz_points(msg_cloud, True)
    # print(rospy.Time.now().to_sec())

    # print(np_p.size)
    scores, dt_box_lidar, types = proc_1.run(np_p)                                                       
    if len(dt_box_lidar) == 0:
        netret = []
    else:
        np_p[:, 2] -= 2.0
        bbox_corners3d = boxes_to_corners_3d(dt_box_lidar)
        corners = bbox_corners3d.reshape((bbox_corners3d.shape[0],-1))
        netret = np.column_stack((dt_box_lidar, scores, types, corners))

    objs = makeobjects(netret)
    lidartracker.update(objs)

    lidarret = {
        'timestamp' : msg.header.stamp.to_sec(),
        'tracklist' : lidartracker.trackerList
    }

    fusionnode.updateLidarRet(lidarret)

    # prediction
    # predinput = PerceptionObstacles() 
    # predinput.header.stamp = msg.header.stamp
    # predinput.header.frame_id = frame_id
    # for trackedobj in lidartracker.trackerList:
    #     obj = PerceptionObstacle()
    #     obj.id = trackedobj.id
    #     obj.type = int(trackedobj.objtype)
    #     obj.x = trackedobj.bbox3d.centerpoint.x
    #     obj.y = trackedobj.bbox3d.centerpoint.y
    #     obj.heading = trackedobj.bbox3d.heading
    #     obj.length = trackedobj.bbox3d.length
    #     obj.width = trackedobj.bbox3d.width
    #     obj.height = trackedobj.bbox3d.height
    #     predinput.obstacles.append(obj)

    # if len(predinput.obstacles) != 0:
    #     pred, mean_xy = predictor.obstacle_callback(predinput)

    empty_markers = MarkerArray()
    clear_marker = Marker()
    clear_marker.header.stamp = msg.header.stamp
    clear_marker.header.frame_id = frame_id
    clear_marker.ns = "tracked objects"
    clear_marker.id = 0
    clear_marker.action = clear_marker.DELETEALL
    clear_marker.lifetime = rospy.Duration()
    empty_markers.markers.append(clear_marker)
    pub_bbox_array.publish(empty_markers)

    bbox_arry = MarkerArray()

    # draw trajectory
    # for i in range(len(lidartracker.trackerList)):
    #     trajectory = Marker()
    #     trajectory.pose = Pose(orientation = Quaternion(x=0,y=0,z=0,w=1))
    #     trajectory.type = Marker.LINE_STRIP
    #     trajectory.ns = "trajectory"
    #     trajectory.lifetime = rospy.Duration()
    #     trajectory.header.stamp = msg.header.stamp
    #     trajectory.header.frame_id = frame_id

    #     pred_xy = pred[0, :, :, i]
    #     for j in range(pred_xy.shape[-1]):
    #         p = Point()
    #         p.x = pred_xy[0, j] + mean_xy[0]
    #         p.y = pred_xy[1, j] + mean_xy[1]
    #         p.z = 0
    #         trajectory.points.append(p)

    #     trajectory.scale.x = 0.1
    #     trajectory.color.a = 1.0
    #     trajectory.color.r = 0.0
    #     trajectory.color.g = 0.0
    #     trajectory.color.b = 1.0
    #     trajectory.id = i+1000
    #     bbox_arry.markers.append(trajectory)


    # draw detection objs
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

    # draw tracked objs
    bboxid = 0
    for tracked in lidartracker.trackerList:
        if tracked.age < -5:
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


def imgret_callback(msg):
    global lastimgmsgtime
    rospy.loginfo('imgret_callback:::msg time: %s, time diff: %s', msg.header.stamp.to_sec(),
    msg.header.stamp.to_sec() - lastimgmsgtime)
    lastimgmsgtime =  msg.header.stamp.to_sec()

    imgret={
        'timestamp':msg.header.stamp.to_sec(),
        'imgretlist':msg.objects
    }

    fusionnode.updateImgRet(imgret)

def gps_callback(msg):
    rospy.loginfo('gps::lat:%s, log:%s', msg.latitude, msg.longitude)


def spin():
    rospy.spin()

if __name__ == "__main__":
    
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
    sub_lidar_topic = [
        "/velodyne_points",
        "/rslidar_points",
        "/kitti/velo/pointcloud",
        "/calibrated_cloud",
        "/segmenter/points_nonground"
    ]
    sub_ = rospy.Subscriber(sub_lidar_topic[1], PointCloud2, lidar_callback, queue_size=1, buff_size=2**24)
    imgretsub = rospy.Subscriber("/img_obj", objArray, imgret_callback, queue_size=1)
    gpssub = rospy.Subscriber("/gps/fix", NavSatFix, gps_callback, queue_size=10)
    pub_bbox_array = rospy.Publisher('lidar_net_results', MarkerArray, queue_size=1)
    # pub_point2_ = rospy.Publisher('lidar_points', PointCloud2, queue_size=1)
    # pub_object_array = rospy.Publisher('lidar_DL_objects', ObjectArray, queue_size=1)

    # print("lidar net ros start!")
    # # rate = rospy.Rate(10)
    # rospy.spin()
    # rate.sleep()

    bagfile = '/space/data/sq/lidar_camera_imu_2021-06-03-10-18-20-4.bag'
    bagfile = '/space/data/sq/lidar0429_3.4.bag'
    bagfile = '/space/data/sq/0525_1_2021-01-21-19-02-39.bag'
    bagfile = '/space/data/sq/zj_lidar_camera_2021-06-15-09-41-15.bag'
    bagfile = '/space/data/sq/cxg/output1.bag'
    # bagfile = '/space/data/sq/rslidar_mems.bag'
    # bagfile = '/space/data/sq/rslidar_mems.bag'
    bag = rosbag.Bag(bagfile,'r')
    print(bag.get_type_and_topic_info())
    # pub_pt = rospy.Publisher('/rslidar_points', PointCloud2, queue_size=1)
    # pub_img = rospy.Publisher('/cam_front', Image, queue_size=1)
    # bag_data = bag.read_messages()
    # for topic, msg, t in bag_data:

    #     # broadcaster = tf2_ros.StaticTransformBroadcaster()
    #     # static_transformStamped = geometry_msgs.msg.TransformStamped()

    #     # static_transformStamped.header.stamp = msg.header.stamp
    #     # static_transformStamped.header.frame_id = "world"
    #     # static_transformStamped.child_frame_id = 'velodyne'

    #     # # lidar pos in world frame
    #     # static_transformStamped.transform.translation.x = 0
    #     # static_transformStamped.transform.translation.y = 0
    #     # static_transformStamped.transform.translation.z = 0

    #     # quat = tf.transformations.quaternion_from_euler(0,0,0)
    #     # static_transformStamped.transform.rotation.x = quat[0]
    #     # static_transformStamped.transform.rotation.y = quat[1]
    #     # static_transformStamped.transform.rotation.z = quat[2]
    #     # static_transformStamped.transform.rotation.w = quat[3]

    #     # broadcaster.sendTransform(static_transformStamped)
    #     # tfBuffer = tf2_ros.Buffer()
    #     # listener = tf2_ros.TransformListener(tfBuffer)

    #     # try:
    #     #     transform = tfBuffer.lookup_transform("world","velodyne", rospy.Time())
    #     #     pointInWorldFrame = do_transform_cloud(msg, transform)
    #     # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #     #     continue

    #     if topic == '/rslidar_points':
    #         pub_pt.publish(msg)
    #         # pub_pt.publish(pointInWorldFrame)
    #         lidar_callback(msg)
    #     # if topic == '/cam_front/csi_cam/image_raw':
    #     if topic == '/wideangle/image_raw':
    #         pub_img.publish(msg)

    spin_thread = threading.Thread(target = spin)
    spin_thread.start()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # rospy.logwarn('inwhile')
        rate.sleep()
