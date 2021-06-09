import rospy
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
from collections import OrderedDict, deque

MAX_HISTORY_TIME = 3.0
FPS = 2.0


class LRUCache:
    def __init__(self, size):
        self.size = size
        self.linked_map = OrderedDict()

    def set(self, key, value):
        if key in self.linked_map:
            self.linked_map.pop(key)
        if self.size == len(self.linked_map):
            self.linked_map.popitem(last=False)
        self.linked_map.update({key: value})

    def get(self, key):
        value = self.linked_map.get(key)
        if value:
            self.linked_map.pop(key)
            self.linked_map.update({key: value})
        return value
    def get_list(self):
        keys = []
        values = []
        for key, value in self.linked_map.items():
            keys.append(key)
            values.append(value)
        return keys, values


class Obstacle():
    def __init__(self, time_stamp, obs):
        self.perception_obs = obs
        self.time_stamp = time_stamp
        self.id = obs.id
        self.type = obs.type
        self.x = obs.x
        self.y = obs.y
        self.heading = obs.heading
        self.feature_history = deque()
        self.feature_history.append([time_stamp, self.x, self.y, self.heading])

    def insert(self, perception_obs, idx, timestamp):
        if len(self.feature_history) < 1:
            return False
        last_timestamp = self.feature_history[-1][0]
        print('last_timestamp', last_timestamp)
        if timestamp < last_timestamp:
            print('earlier')
            return False
        if self.id != idx:
            print('mismatch id')
            return False
        if self.type != perception_obs.type:
            print('mismatch type')
            return False
        if (timestamp - last_timestamp) >= rospy.Duration(1.0 / FPS):
            self.perception_obs = perception_obs
            self.time_stamp = timestamp
            self.id = perception_obs.id
            self.type = perception_obs.type
            self.x = perception_obs.x
            self.y = perception_obs.y
            self.heading = perception_obs.heading
            txy = [timestamp, perception_obs.x, perception_obs.y, perception_obs.heading]
            self.feature_history.append(txy)
            # discard outdated feature
        # else:
        #     print('sampling....', timestamp)
        self.discard_history()

    def discard_history(self):
        latest_timestamp = self.feature_history[-1][0]
        while latest_timestamp - self.feature_history[0][0] > rospy.Duration(MAX_HISTORY_TIME):
            self.feature_history.popleft()


class ObstacleContainer():
    def __init__(self, max_obs_num):
        self.max_obs_num = max_obs_num
        self.time_stamp = rospy.Time(0.0)
        self.obstacles = LRUCache(max_obs_num) # key:id value:Obstacle

    def get_obstacle_lru_update(self, obj_id):
        value = self.obstacles.get(obj_id)
        return value

    def insert(self, stamp, objs):
        if stamp < self.time_stamp:
            return
        self.time_stamp = stamp
        # insert obstacles one by one
        for obj in objs:
            obj_id = obj.id
            obs = self.get_obstacle_lru_update(obj_id)
            if obs:
                obs.insert(obj, obj_id, self.time_stamp)
                print('refresh obs: ', obj_id)
            else:
                print('new obs: ', obj_id)
                obs = Obstacle(self.time_stamp, obj)
                self.obstacles.set(obj_id, obs)
    def valid_obstacles(self, frame_num):
        # obstacles history size >= frame_num
        valid_obstacles = []
        invalid_obstacles = []
        _, obstacles = self.obstacles.get_list()
        for obs in obstacles:
            if len(obs.feature_history) >= frame_num:
                valid_obstacles.append(obs)
            else:
                invalid_obstacles.append(obs)
        return valid_obstacles, invalid_obstacles

