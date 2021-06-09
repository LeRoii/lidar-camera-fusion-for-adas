import rospy
import sys
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
import math
import argparse
import numpy as np

class ObsSender():
    def __init__(self, file_name, dt):
        self.dt = dt
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.w_left = 0.0
        self.w_right = 0.0
        self.speed = 0.0
        self.yawrate = 0.0
        self.obstacles = []
        data = np.genfromtxt(file_name, delimiter=" ")
        for obs_data in data:
            obs_m = PerceptionObstacle()
            # object_id, object_type, position_x, position_y, heading, object_length, object_width, object_height, velocity
            obs_m.id = int(obs_data[0])
            obs_m.type = int(obs_data[1]) # 0vehicle,  1pedestrian, 2cyclist, 3other
            obs_m.x = obs_data[2]
            obs_m.y = obs_data[3]
            obs_m.heading = obs_data[4]
            obs_m.length = obs_data[5]
            obs_m.width = obs_data[6]
            obs_m.height = obs_data[7]
            obs_m.velocity = obs_data[8]
            self.obstacles.append(obs_m)


    def run(self):
        rate = self.dt
        rospy.loginfo('rate = %f', rate)
        obspub = rospy.Publisher(
            "/perception/obstacles", PerceptionObstacles, queue_size=10)
        while not rospy.is_shutdown():
            obsmsg = PerceptionObstacles()
            obsmsg.header.frame_id = "base_link"
            obsmsg.header.stamp = rospy.Time.now()
            obsmsg.obstacles = []
            for obs_m in self.obstacles:
                obs_m.x += obs_m.velocity * math.cos(obs_m.heading) * self.dt
                obs_m.y += obs_m.velocity * math.sin(obs_m.heading) * self.dt
                obsmsg.obstacles.append(obs_m)
                obspub.publish(obsmsg)
            if rate:
                rospy.sleep(rate)
            else:
                rospy.sleep(1.0)


# Main function.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obs_sender')
    parser.add_argument('filename', type=str,
                        help='a description file of obstacles')
    parser.add_argument('hz', type=int,
                        help='hz of rosnode')
    args = parser.parse_args()
    print('send obstacles according to: ', args.filename)
    rospy.init_node('obs_sender')
    rate = 1.0 / float(args.hz)
    obs_sender = ObsSender(args.filename, rate)
    try:
        obs_sender.run()
    except rospy.ROSInterruptException:
        pass
