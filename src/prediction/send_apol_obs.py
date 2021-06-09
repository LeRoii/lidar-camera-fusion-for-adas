import rospy
import sys
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
import math
import argparse
import numpy as np

def get_type(apol_t):
    # 0----vehicle,  
    # 1----pedestrian,
    # 2----cyclist, 
    # 3----other
    if apol_t < 3:
        return 0
    elif apol_t == 3:
        return 1
    elif apol_t == 4:
        return 2
    else:
        return 3


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
        self.data = np.genfromtxt(file_name, delimiter=" ")
        self.frame_ids = np.unique(self.data[:, 0]).tolist()
        print('totalframe:', len(self.frame_ids))

    def run(self):
        rate = self.dt
        rospy.loginfo('rate = %f', rate)
        obspub = rospy.Publisher(
            "/perception/obstacles", PerceptionObstacles, queue_size=10)
        i = 0
        FPS = int(0.5 / rate)
        while not rospy.is_shutdown():
            frame_id = self.frame_ids[int(i / FPS)  % len(self.frame_ids)]
            print('frame id: ', frame_id)
            obsmsg = PerceptionObstacles()
            obsmsg.header.frame_id = "base_link"
            obsmsg.header.stamp = rospy.Time.now()
            obsmsg.obstacles = []
            sequence_data = self.data[np.where(self.data[:, 0] == frame_id)]
            for obs_data in sequence_data:
                obs_m = PerceptionObstacle()
                # frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading
                obs_m.id = int(obs_data[1])
                obs_m.type = get_type(obs_data[2])
                obs_m.x = obs_data[3]
                obs_m.y = obs_data[4]
                obs_m.heading = obs_data[9]
                obs_m.length = obs_data[6]
                obs_m.width = obs_data[7]
                obs_m.height = obs_data[8]
                obs_m.velocity = 0.0
                obsmsg.obstacles.append(obs_m)
            obspub.publish(obsmsg)
            i += 1
            if rate:
                rospy.sleep(rate)
            else:
                rospy.sleep(1.0)


# Main function.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='obs_sender')
    parser.add_argument('filename', type=str,
                        help='a apolloscape format description file of obstacles')
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
