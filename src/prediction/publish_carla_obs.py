#coding=utf-8
import rospy
from derived_object_msgs.msg import ObjectArray, Object
from obs_msgs.msg import PerceptionObstacle, PerceptionObstacles
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
import tf
import tf2_geometry_msgs
import tf_conversions
import math

def get_type(calssification):
    # if calssification < 4:
    #     return 3
    # elif calssification == 4:
    #     return 1
    # elif calssification == 5 or calssification == 8:
    #     return 2
    # elif calssification < 10:
    #     return 0
    # else:
    return 0

M_PI_2 = math.pi*2

class CarlaSender():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.obspub = rospy.Publisher(
            "/perception/obstacles", PerceptionObstacles, queue_size=10)
        self.locpub = rospy.Publisher(
            "/localization/pose", Pose, queue_size=10)
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()

    def convert_lidar_to_world(self, data):
        
        # // 障碍物中心位置、速度、置信度
        data.pose.position.x = data.pose.position.x
        data.pose.position.y = data.pose.position.y + 1.16
        data.pose.position.z = data.pose.position.z + 1.66

        # (_, _, lidar_yaw) = tf_conversions.transformations.euler_from_quaternion(
        #     data.pose.orientation)

        orientation = (data.pose.orientation.x, data.pose.orientation.y,
                        data.pose.orientation.z, data.pose.orientation.w)
        (_, _, lidar_yaw) = tf_conversions.transformations.euler_from_quaternion(
            orientation)

        lidar_yaw = -lidar_yaw - M_PI_2 + self.yaw
        # print(self.yaw)

        rotate_angle = self.yaw - M_PI_2

        data.pose.position.x = data.pose.position.x * math.cos(rotate_angle) - data.pose.position.y * math.sin(rotate_angle)
        data.pose.position.y = data.pose.position.x * math.sin(rotate_angle) + data.pose.position.y * math.cos(rotate_angle)

        data.pose.position.x = data.pose.position.x + self.x
        data.pose.position.y = data.pose.position.y + self.y

        # // speed convert from car to map
        vx = data.twist.linear.x 
        vy = data.twist.linear.y
        vx = vx * math.cos(rotate_angle) - vy * math.sin(rotate_angle)
        vy = vx * math.sin(rotate_angle) + vy * math.cos(rotate_angle)

        data.twist.linear.x = vx + self.vx
        data.twist.linear.y = vy + self.vy
        obj_heading = math.atan2(data.twist.linear.y, data.twist.linear.x)
        orientation = tf_conversions.transformations.quaternion_from_euler(0,0,obj_heading)
        data.pose.orientation.x = orientation[0]
        data.pose.orientation.y = orientation[1]
        data.pose.orientation.z = orientation[2]
        data.pose.orientation.w = orientation[3]

        return data
        

    def obj_callback(self, data):
        obsmsg = PerceptionObstacles()
        obsmsg.header.frame_id = "world"
        obsmsg.header.stamp = rospy.Time.now()
        obsmsg.obstacles = []
        self.listener.waitForTransform(
            'base_link', 'world', rospy.Time(), rospy.Duration(4.0))
        try:
            self.listener.lookupTransform(
                'base_link', 'world', rospy.Time(0))
            transform = self.listener.getLatestCommonTime(
                "base_link", "world")
            for obj in data.objects:
                distance = ((obj.pose.position.x - self.x) * (obj.pose.position.x - self.x) +
                            (obj.pose.position.y - self.y) * (obj.pose.position.y - self.y))**0.5
                                
                if distance < 100.0:
                    # print(distance,obj.id)
                    obs_m = PerceptionObstacle()
                    obs_m.id = obj.id
                    # 0vehicle,  1pedestrian, 2cyclist, 3other
                    obs_m.type = get_type(obj.classification)
                    # 从全局坐标系转成车体坐标系
                    pose_in_world = PoseStamped()
                    pose_in_world.header.frame_id = "world"
                    pose_in_world.pose = obj.pose

                    pose_in_base = self.listener.transformPose(
                        "base_link", pose_in_world)
                    obs_m.x = pose_in_world.pose.position.x
                    obs_m.y = pose_in_world.pose.position.y
                    orientation = (pose_in_world.pose.orientation.x, pose_in_world.pose.orientation.y,
                                pose_in_world.pose.orientation.z, pose_in_world.pose.orientation.w)
                    (_, _, yaw) = tf_conversions.transformations.euler_from_quaternion(
                        orientation)


                    obs_m.heading = yaw
                    obs_m.length = obj.shape.dimensions[0]
                    obs_m.width = obj.shape.dimensions[1]
                    obs_m.height = obj.shape.dimensions[2]
                    obs_m.velocity = 0.0
                    obsmsg.obstacles.append(obs_m)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("!!")
        self.obspub.publish(obsmsg)

    def obj_callback_real(self, data):
        # quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        # self.br.sendTransform((0, 0, 0), 
        #         (quaternion[0], quaternion[1], quaternion[2], quaternion[3]), 
        #         rospy.Time.now(), "base_link", "world")

        obsmsg = PerceptionObstacles()
        obsmsg.header.frame_id = "map"
        obsmsg.header.stamp = rospy.Time.now()
        obsmsg.obstacles = []
        # self.listener.waitForTransform(
        #     'base_link', 'world', rospy.Time(), rospy.Duration(4.0))
        try:
            # self.listener.lookupTransform(
            #     'base_link', 'world', rospy.Time(0))
            # transform = self.listener.getLatestCommonTime(
            #     "base_link", "world")
            for obj in data.objects:

                distance = math.sqrt(obj.pose.position.x**2+obj.pose.position.y**2)

                # print(distance)
                if distance < 100.0:
                    obj = self.convert_lidar_to_world(obj)
                    obs_m = PerceptionObstacle()
                    obs_m.id = obj.id
                    # 0vehicle,  1pedestrian, 2cyclist, 3other
                    obs_m.type = get_type(obj.classification)
                    obs_m.x = obj.pose.position.x
                    obs_m.y = obj.pose.position.y
                    # print(obs_m.x,obs_m.y)
                    orientation = (obj.pose.orientation.x, obj.pose.orientation.y,
                                   obj.pose.orientation.z, obj.pose.orientation.w)
                    (_, _, yaw) = tf_conversions.transformations.euler_from_quaternion(
                        orientation)
                    obs_m.heading = yaw
                    obs_m.length = obj.shape.dimensions[0]
                    obs_m.width = obj.shape.dimensions[1]
                    obs_m.height = obj.shape.dimensions[2]
                    obs_m.velocity = 0.0
                    obsmsg.obstacles.append(obs_m)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("!!")
        
        # print((obsmsg.obstacles))
        self.obspub.publish(obsmsg)

    # 获取自车状态信息
    def loc_callback(self, data):


        self.locpub.publish(data.pose.pose)
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

        print(self.x, self.y)

        self.vx = data.twist.twist.linear.x
        self.vy = data.twist.twist.linear.y

        orientation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        (_, _, self.yaw) = tf_conversions.transformations.euler_from_quaternion(
            orientation)

        # 发布全局坐标系和以车辆后轴中心为原点的车体坐标系的tf
        self.br.sendTransform((self.x, self.y, data.pose.pose.position.z), 
                (data.pose.pose.orientation.x, data.pose.pose.orientation.y,                                                                      
                data.pose.pose.orientation.z, data.pose.pose.orientation.w), 
                rospy.Time.now(), "base_link", "world")

    # 获取自车状态信息
    def loc_callback_real(self, data):
        self.locpub.publish(data.pose.pose)
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

        self.vx = data.twist.twist.linear.x
        self.vy = data.twist.twist.linear.y

        orientation = (data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        (_, _, self.yaw) = tf_conversions.transformations.euler_from_quaternion(
            orientation)
        
        # 发布全局坐标系和以车辆后轴中心为原点的车体坐标系的tf
        # self.br.sendTransform((self.x, self.y+1.16, data.pose.pose.position.z+1.66), 
        #         (data.pose.pose.orientation.x, data.pose.pose.orientation.y,                                                                      
        #         data.pose.pose.orientation.z, data.pose.pose.orientation.w), 
        #         rospy.Time.now(), "base_link", "world")

    def run(self):
        rospy.Subscriber("/carla/objects",
                         ObjectArray, self.obj_callback)
        rospy.Subscriber("/carla/ego_vehicle/odometry",
                         Odometry, self.loc_callback)

        rospy.Subscriber("/discovery/objects_to_plan",  ObjectArray, self.obj_callback_real)
        rospy.Subscriber("/discovery/location/pose",  Odometry, self.loc_callback_real) 

        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('carla_to_per')
    carla_sender = CarlaSender()
    try:
        carla_sender.run()
    except rospy.ROSInterruptException:
        pass
