import numpy as np
import torch
import rospy
import argparse
import math
from collections import deque
from scipy import spatial
from model import Model
from layers.graph import Graph

neighbor_distance = 10  # meter
history_frame_num = 6
pred_frame_num = 6
dev = 'cuda:0'


class Obstacle():
    def __init__(self, time_stamp, oid, otype, ox, oy, oheading):
        self.time_stamp = time_stamp
        self.id = oid
        self.type = otype
        self.x = ox
        self.y = oy
        self.heading = oheading
        self.feature_history = deque()
        self.feature_history.append([time_stamp, self.x, self.y, self.heading])

    def insert(self, ox, oy, oheading, timestamp):
        if len(self.feature_history) < 1:
            return False
        last_timestamp = self.feature_history[-1][0]
        if timestamp < last_timestamp:
            print('earlier')
            return False
        if (timestamp - last_timestamp) >= 0.5:
            self.time_stamp = timestamp
            self.x = ox
            self.y = oy
            self.heading = oheading
            txy = [timestamp, ox, oy, oheading]
            self.feature_history.append(txy)
            # discard outdated feature
        # else:
        #     print('sampling....', timestamp)
        self.discard_history()

    def discard_history(self):
        latest_timestamp = self.feature_history[-1][0]
        while latest_timestamp - self.feature_history[0][0] > 3.0:
            self.feature_history.popleft()


checkpoint = torch.load('./data/model_epoch_0016.pt')
graph_args = {'max_hop': 2, 'num_node': 120}
graph = Graph(**graph_args)
model = Model(in_channels=4, graph_args=graph_args,
              edge_importance_weighting=True)
model.to(dev)
model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
model.eval()


des_data = np.genfromtxt('./description.txt', delimiter=" ")
frame_ids = np.unique(des_data[:, 0]).tolist()
all_ids = np.unique(des_data[:, 1]).tolist()
sequence_data = des_data[np.where(des_data[:, 0] == frame_ids[-1])]
valid_ids = []
last_xy = []
for obs_data in sequence_data:
    valid_ids.append(obs_data[1])
    last_xy.append([obs_data[3], obs_data[4]])
xy = np.array(last_xy)
# print('xy: ', xy)
m_xy = np.mean(xy, axis=0)
non_visible_ids = list(set(all_ids) - set(valid_ids))
frame_feature = []
for frame_id in frame_ids:
    # print('frame id: ', frame_id)
    sequence_data = des_data[np.where(des_data[:, 0] == frame_id)]
    frame_obs_ids = np.unique(des_data[:, 1]).tolist()
    obstacle_feature = []
    for oid in (valid_ids + non_visible_ids):
        print('id: ', oid)
        obs_data = sequence_data[np.where(sequence_data[:,1]==oid)]
        if obs_data.size != 0:
            if oid in valid_ids:
                obstacle_feature.append([obs_data[0][3] -m_xy[0], obs_data[0][4]-m_xy[1], obs_data[0][9], 1])
            else:
                obstacle_feature.append([obs_data[0][3]-m_xy[0], obs_data[0][4]-m_xy[1], obs_data[0][9], 0])
        else:
            obstacle_feature.append([0, 0, 0, 0])
    frame_feature.append(obstacle_feature)

obstacle_feature_array = np.array(frame_feature) # tvc
obs_num = len(valid_ids)
# print('obstacle_feature_array: ', obstacle_feature_array)
# compute distance between any pair of two objects
dist_xy = spatial.distance.cdist(xy, xy)
neighbor_matrix = np.zeros((120, 120))
neighbor_matrix[:obs_num, :obs_num] = (dist_xy < neighbor_distance).astype(int)
# zero_centralize

feature_array = obstacle_feature_array # TVC
feature_shape = feature_array.shape
print('valid id： ', valid_ids)
print('feature_list: ', feature_array.shape)
rescale_xy = torch.ones((1,2,1,1)).to(dev)

# feature_array = np.concatenate((feature_list, np.ones((feature_shape[0], feature_shape[1], 1))), axis = 2)
# # to nctv
feature = np.zeros(
    (1, feature_array.shape[2], feature_array.shape[0], 120))
feature[:,:,:,:feature_array.shape[1]]= np.array(
    [np.transpose(feature_array, (2, 0, 1))])

ori_output_last_loc = feature[:, :2,
                              history_frame_num-1: history_frame_num, :]
ori_output_last_loc_t = torch.from_numpy(ori_output_last_loc).float().to(dev)
print('ori_output_last_loc', ori_output_last_loc_t[0,:,0,:])
# get velocity
data = torch.from_numpy(feature).detach()
new_mask = (data[:, : 2, 1:] != 0) * (data[:, : 2, : -1] != 0)
data[:, : 2, 1:] = (data[:, : 2, 1:] - data[:, : 2, : -1]
                    ).float() * new_mask.float()
data[:, :2, 0] = 0

feature_data = data.float().to(dev)
feature_data[:,:2] = feature_data[:,:2]

now_adjacency = graph.get_adjacency(neighbor_matrix)
now_A = np.array([graph.normalize_adjacency(now_adjacency)])
A = torch.from_numpy(now_A).float().to(dev)
# print(feature_data[0, :, 0, :])
# print(A[0, 0, :, :])
predicted = model(pra_x=feature_data, pra_A=A, pra_pred_length=pred_frame_num,
                  pra_teacher_forcing_ratio=0, pra_teacher_location=None)

for ind in range(1, predicted.shape[-2]):
    predicted[:, :, ind] = torch.sum(predicted[:, :, ind-1:ind+1], dim=-2)
# print('predicted: ', predicted[0,:,0,:])
predicted += ori_output_last_loc_t

now_pred = predicted.detach().cpu().numpy()  # (N, C, T, V)=(N, 2, 6, 120)
for i in range(obs_num):
    obj_id = valid_ids[i]
    print(obj_id)
    pred_xy = now_pred[0, :, :, i]
    for j in range(pred_xy.shape[-1]):
        print(pred_xy[0, j] + m_xy[0])
        print(pred_xy[1, j] + m_xy[1])
