from objmsg.msg import obj,objArray
import rospy
from collections import deque
import threading

class fusion():
    def __init__(self):
        self.lidarmsgque = deque(maxlen=10)
        self.imgmsgque = deque(maxlen=15)
        self.mutex = threading.Lock()
        pass

    def updateImgRet(self, imgret):
        rospy.logwarn('updateImgRet:::msg time: %s', imgret['timestamp'])
        self.mutex.acquire()
        self.imgmsgque.append(imgret)
        self.mutex.release()

    def updateLidarRet(self, lidarret):
        rospy.logwarn('updateLidarRet:::msg time: %s', lidarret['timestamp'])
        self.mutex.acquire()
        self.lidarmsgque.append(lidarret)
        self.mutex.release()

    def dequeTimeSync(self):
        if len(self.lidarmsgque) > 0 and len(self.imgmsgque) > 0:
            if self.lidarmsgque[0]['timestamp'] > self.imgmsgque[-1]['timestamp']: #lidar needs to wait img
                self.imgmsgque.clear()
            elif self.lidarmsgque[-1]['timestamp'] < self.imgmsgque[0]['timestamp']: #img needs to wait lidar
                self.lidarmsgque.clear()

    def findMatchedImg(self, timestamp):
        for msg in self.imgmsgque:
            if abs(timestamp - msg['timestamp']) < 0.2:
                return msg
        return None

    def take(self, lidartime):
        msg = self.findMatchedImg(lidartime) 
        if msg is not None:
            print('time diff:',lidartime - msg['timestamp'])