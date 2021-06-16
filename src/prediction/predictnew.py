import numpy as np
import torch
import rospy
import argparse
import math
import time
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
from obs_msgs.msg import TrajPoint, Trajectory
from obs_msgs.msg import PredictedObstacle, PredictedObstacles
from obstacle_container import Obstacle, ObstacleContainer
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from scipy import spatial
from model import Model
from layers.graph import Graph

import glob
import os
import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla

neighbor_distance = 10  # meter
history_frame_num = 6
pred_frame_num = 6
dev = 'cuda:0'

def get_type(calssification):
    if calssification < 4:
        return 3
    elif calssification == 4:
        return 1
    elif calssification == 5 or calssification == 8:
        return 2
    elif calssification < 10:
        return 0
    else:
        return 0

M_PI_2 = math.pi*2

class Predictor(object):

    def __init__(self, obs_num):
        self.pred_pub = rospy.Publisher(
            "/prediction/obstacles", PredictedObstacles, queue_size=10)

        # self.pred_carla = rospy.Publisher(
        #     "/carla/ego_vehicle/waypoints", Path, queue_size=10)

        self.max_obs_num = obs_num
        self.frame_id = 0
        self.container = ObstacleContainer(obs_num)

        checkpoint = torch.load('/space/code/sqadas/rosws/src/prediction/data/model_epoch_0016.pt')
        graph_args = {'max_hop': 2, 'num_node': obs_num}
        self.graph = Graph(**graph_args)
        self.model = Model(in_channels=4, graph_args=graph_args,
                           edge_importance_weighting=True)
        self.model.to(dev)
        self.model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
        self.model.eval()

    def obstacle_callback(self, data):
        # update
        # print('frame_id: ', self.frame_id)
        self.container.insert(
            data.header.stamp, self.frame_id, data.obstacles)
        feature, neighbor_matrix, mean_xy = self.container.get_data(
            (self.frame_id - history_frame_num + 1), self.frame_id)
        # data preparation for GRIP
        start = time.time()
        pred = self.grip_predict(feature, neighbor_matrix)
        print('object number: ',len(data.obstacles),',spend: ', time.time() - start)
        self.publish(data.obstacles, pred, mean_xy)
        self.frame_id += 1

        return pred, mean_xy

    def grip_predict(self, feature, neighbor_matrix):
        last_loc = feature[:, :2, history_frame_num-1:history_frame_num, :]
        ori_output_last_loc = torch.from_numpy(
            last_loc).detach().float().to(dev)
        # get velocity
        data = torch.from_numpy(feature).detach()
        new_mask = (data[:, : 2, 1:] != 0) * (data[:, : 2, : -1] != 0)
        data[:, : 2, 1:] = (data[:, : 2, 1:] - data[:, : 2, : -1]
                            ).float() * new_mask.float()
        data[:, :2, 0] = 0
        feature_data = data.float().to(dev)
        now_adjacency = self.graph.get_adjacency(neighbor_matrix)
        now_A = np.array([self.graph.normalize_adjacency(now_adjacency)])
        A = torch.from_numpy(now_A).float().to(dev)
        predicted = self.model(pra_x=feature_data, pra_A=A, pra_pred_length=pred_frame_num,
                               pra_teacher_forcing_ratio=0, pra_teacher_location=None)
        for ind in range(1, predicted.shape[-2]):
            predicted[:, :, ind] = torch.sum(
                predicted[:, :, ind-1:ind+1], dim=-2)
        predicted += ori_output_last_loc
        now_pred = predicted.detach().cpu().numpy()  # (N, C, T, V)=(N, 2, 6, 120)
        return now_pred
    
    def publish(self, obstacles, pred, mean_xy):
        pred_obstacles = PredictedObstacles()
        ego_path = Path()
        pred_obstacles.header.frame_id = "map"
        pred_obstacles.header.stamp = rospy.Time.now()
        ego_path.header = pred_obstacles.header

        
        # speed_file = 'speed.txt'
        # with open(speed_file, 'w') as writer:

        for i in range(len(obstacles)):
            pred_obstacle = PredictedObstacle()
            obj = obstacles[i]
            pred_obstacle.perception = obj
            pred_traj = Trajectory()
            # c_point = TrajPoint()
            # c_point.time = 0.0
            # c_point.x = obj.x
            # c_point.y = obj.y            
            # result = 'id = {},time = {},x = {},y = {},'.format(
            # obj.id,c_point.time,c_point.x,c_point.y) + '\n'
            # writer.write(result)
            # pred_traj.points.append(c_point)
            pred_xy = pred[0, :, :, i]
            for j in range(pred_xy.shape[-1]):
                point = TrajPoint()
                point.time = float(j+1) * 0.5
                point.x = pred_xy[0, j] + mean_xy[0]
                point.y = pred_xy[1, j] + mean_xy[1]
                pred_traj.points.append(point)
                # result = 'id = {},time = {},x = {},y = {},'.format(
                # obj.id,point.time,point.x,point.y) + '\n'
                # writer.write(result)

            pred_obstacle.traj = pred_traj
            pred_obstacles.predicted_obstacles.append(pred_obstacle)

        # for obj_trj in pred_obstacles.predicted_obstacles:

        #     # print(obj_trj.perception.type)
        #     car_type = (obj_trj.perception.type)

        #     for index,point in enumerate(obj_trj.traj.points):
        #         if index==0:
        #             continue
        #         begin = carla.Location()
        #         end= carla.Location()
        #         begin.x = obj_trj.traj.points[index].x
        #         begin.y = -obj_trj.traj.points[index].y
        #         begin.z = 0.2

        #         end.x = obj_trj.traj.points[index-1].x
        #         end.y = -obj_trj.traj.points[index-1].y
        #         end.z = 0.2
        #         if car_type ==3:
        #             car_color = carla.Color(r=0, g=255, b=0)
        #         elif car_type ==2:
        #             car_color = carla.Color(r=255, g=0, b=0)
        #         if car_type ==1:
        #             car_color = carla.Color(r=0, g=0, b=255)
        #         else:
        #             car_color = carla.Color(r=255, g=0, b=0)


        #         self.world.debug.draw_line(begin, end, thickness=0.2,color= car_color,life_time=0.12)

                # size=0.05,

                # self.world.debug.draw_point(begin, size=0.5,
                #                                 color=carla.Color(r=255, g=0, b=0),
                #                                 life_time= 0.1)

        self.pred_pub.publish(pred_obstacles)
    def run(self):
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()  
        rospy.Subscriber("/perception/obstacles",
                         PerceptionObstacles, self.obstacle_callback)
        rospy.spin()
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prediction node')
    parser.add_argument('--max_obs_num', type=int,default=120,
                        help='max num of obstacles per frame')
    args = parser.parse_args()
    rospy.init_node('predictor')
    predictor = Predictor(args.max_obs_num)
    try:
        predictor.run()
    except rospy.ROSInterruptException:
        pass
