import rospy
import numpy as np
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
from collections import deque, OrderedDict
from scipy import spatial

MAX_HISTORY_TIME = 3.0

# Least Recently Used
class LRUCache:
    '''
    最近最少使用队列实现,最近使用的键值放后面
    '''
    def __init__(self, size):
        self.size = size
        self.linked_map = OrderedDict()

    def set(self, key, value):
        if key in self.linked_map:
            # pop() 弹出指定key值
            self.linked_map.pop(key)
        if self.size == len(self.linked_map):
            '''
            popitem(last=True) last出去，根据LIFO原则，后进先出
            popitem(last=False) last不出去，根据FIFO原则，先进先出
            '''
            self.linked_map.popitem(last=False)
        #将key及value键入 
        self.linked_map.update({key: value})

    def get(self, key):
        value = self.linked_map.get(key)
        if value:
            self.linked_map.pop(key)
            self.linked_map.update({key: value})
        return value
    
    def get_silently(self, key):
        return self.linked_map.get(key)


class Obstacle():
    def __init__(self, time_stamp, frame_id, obs):

        self.perception_obs = obs
        self.time_stamp = time_stamp
        self.id = obs.id
        self.type = obs.type
        self.x = obs.x
        self.y = obs.y
        self.heading = obs.heading
        self.frame_id = frame_id
        self.feature_history = deque()
        self.feature_history.append([time_stamp, frame_id, self.x, self.y, self.heading])

    def insert(self, perception_obs, idx, timestamp, frameid):
        if len(self.feature_history) < 1:
            return False
        last_timestamp = self.feature_history[-1][0]
        # print('last_timestamp', last_timestamp)
        if timestamp < last_timestamp:
            print('earlier')
            return False
        if self.id != idx:
            print('mismatch id')
            return False
        if self.type != perception_obs.type:
            print('mismatch type')
            return False
        self.perception_obs = perception_obs
        self.time_stamp = timestamp
        self.id = perception_obs.id
        self.type = perception_obs.type
        self.x = perception_obs.x
        self.y = perception_obs.y
        self.heading = perception_obs.heading
        self.frame_id = frameid
        txy = [timestamp, frameid, perception_obs.x,
                perception_obs.y, perception_obs.heading]
        self.feature_history.append(txy)
        # discard outdated feature
        self.discard_history()

    def discard_history(self):
        latest_timestamp = self.feature_history[-1][0]
        while latest_timestamp - self.feature_history[0][0] > rospy.Duration(MAX_HISTORY_TIME):
            # 进入队列，默认从右边进入，所以左边为最远的时间
            self.feature_history.popleft()
    
    def start_frame(self):
        return self.feature_history[0][1]
    
    def end_frame(self):
        return self.feature_history[-1][1]

    def get_frame_list(self):
        f_list = []
        for feat in self.feature_history:
            f_list.append(feat[1])
        return f_list

    def get_frame_feature(self, frame_id):
        if frame_id < self.start_frame() or frame_id > self.end_frame():
            return None
        else:
            for feature in self.feature_history:
                if feature[1] == frame_id:
                    return feature
        return None

'''LRU of Obstacle'''
'''key:id value:Obstacle'''
class Obstacles(LRUCache):
    def get_valid_ids(self, frame_list):
        valid_ids = []
        ids = self.linked_map.keys()
        for oid in ids:
            obs = self.linked_map.get(oid)
            obs_frame_list = obs.get_frame_list()
            # intersection() 方法用于返回两个或更多集合中都包含的元素，即交集
            if set(obs_frame_list).intersection(set(frame_list)):
                valid_ids.append(oid)
        return valid_ids


class ObstacleContainer():
    def __init__(self, max_obs_num):
        self.max_obs_num = max_obs_num
        self.time_stamp = rospy.Time(0.0)
        self.obstacles = Obstacles(self.max_obs_num)
        self.visible_ids = []

    def get_obstacle_lru_update(self, obj_id):
        value = self.obstacles.get(obj_id)
        return value

    def insert(self, stamp, frame_id, objs):
        if stamp < self.time_stamp:
            return
        self.time_stamp = stamp
        self.visible_ids = []
        for obj in objs:
            obj_id = obj.id
            self.visible_ids.append(obj_id)
            obs = self.get_obstacle_lru_update(obj_id)
            if obs:
                obs.insert(obj, obj_id, self.time_stamp, frame_id)
                # print('refresh obs: ', obj_id)
            else:
                # print('new obs: ', obj_id)
                obs = Obstacle(self.time_stamp, frame_id, obj)
                self.obstacles.set(obj_id, obs)
    def get_data(self, start, end):
        start = end + 1 - 30
        # 激光一秒十帧，取两帧
        sample_frame_list = [f for f in range(start, end+1, 5)]
        # feature, mean_xy, neighbor_matrix
        # NCTV
        all_ids = self.obstacles.get_valid_ids(sample_frame_list)
        # print('visible ids: ', self.visible_ids)
        non_visible_ids = list(set(all_ids) - set(self.visible_ids))
        last_xy = []
        for vid in self.visible_ids:
            value = self.obstacles.linked_map.get(vid).feature_history
            last_xy.append([value[-1][2], value[-1][3]])
        xy = np.array(last_xy)
        m_xy = np.mean(xy, axis=0)
        # print('mean: ', m_xy)
        dist_xy = spatial.distance.cdist(xy, xy)
        neighbor_matrix = np.zeros((120, 120))
        obs_num = len(self.visible_ids)
        neighbor_matrix[:obs_num, :obs_num] = (dist_xy < 10.0).astype(int)
        frame_list = []
        # 对于每一帧，获取特征
        for i in sample_frame_list:
            frame_feaure = []
            for oid in (self.visible_ids + non_visible_ids):
                obs = self.obstacles.get_silently(oid)
                obs_frame_data = obs.get_frame_feature(i)
                if obs_frame_data:
                    if oid in self.visible_ids:
                        frame_feaure.append([obs_frame_data[2]-m_xy[0], obs_frame_data[3]-m_xy[1], obs_frame_data[4], 1]) # x, y, heading, mask
                    else:
                        frame_feaure.append([obs_frame_data[2]-m_xy[0], obs_frame_data[3]-m_xy[1], obs_frame_data[4], 0]) 
                else:
                    # print('00000')
                    frame_feaure.append([0, 0, 0, 0])
            frame_list.append(frame_feaure)
        feature_array = np.array(frame_list) # tvc
        feature = np.zeros((1, feature_array.shape[2], feature_array.shape[0], self.max_obs_num))
        feature[:,:,:,:feature_array.shape[1]]= np.array([np.transpose(feature_array, (2, 0, 1))])
        return feature, neighbor_matrix, m_xy

            
            

                




