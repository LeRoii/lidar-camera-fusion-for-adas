import attr
from abc import ABC, abstractmethod
# from derived_object_msgs.msg import  Object
from pyquaternion import Quaternion
import numpy as np
@attr.s
class point2d():
    x = attr.ib(type=float, default=.0)
    y = attr.ib(type=float, default=.0)

@attr.s
class point3d():
    x = attr.ib(type=float, default=.0)
    y = attr.ib(type=float, default=.0)
    z = attr.ib(type=float, default=.0)

@attr.s
class boundbox2d():
    topleft = attr.ib(type=point2d, default=(0,0))
    bottomright = attr.ib(type=point2d, default=(0,0))

@attr.s
class boundbox3d():
    centerpoint = attr.ib(type=point3d, default=(0,0,0))
    length = attr.ib(type=float, default=.0)
    width = attr.ib(type=float, default=.0)
    height = attr.ib(type=float, default=.0)
    heading = attr.ib(type=Quaternion, default=(0,0,0,0))
    corners = attr.ib(type=list, default=[])
    theta = attr.ib(type=float, default=.0)

@attr.s
class objectbase():
    objtype = attr.ib(type=int, default=0)
    bbox3d = attr.ib(type=boundbox3d, default=boundbox3d())
    bbox2d = attr.ib(type=boundbox2d, default=boundbox2d())
    confidence = attr.ib(type=float, default=.0)
    # age = attr.ib(type=int, default=0)

    # @abstractmethod
    # def fct(self):
    #     pass

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

def makeobjects(networkret):
    objs = []
    for objdata in networkret:
        if objdata[7] < 0.6:    #confidence
            continue
        centerpt = point3d(objdata[0], objdata[1], objdata[2])
        length = objdata[3]
        w = objdata[4]
        h = objdata[5]
        theta = objdata[6]
        if theta < 0: theta += np.pi * 2
        heading = yaw2quaternion(objdata[6])
        corners = []
        for i in range(8):
            pt = point3d(objdata[9+i*3], objdata[10+i*3], objdata[11+i*3])
            corners.append(pt)

        bbox3d = boundbox3d(centerpt,length,w,h,heading,corners,theta)
        obj = objectbase(objtype=objdata[8],bbox3d=bbox3d,confidence=objdata[7])
        objs.append(obj)
    return objs

if __name__=='__main__':
    ret = [[0,0,0,1,1,1,2,0.8,3],[0,0,0,1,1,1,2,0.8,3]]
    obj = makeobjects(ret)
    # p2 = p1 = point2d(2,3)
    # p2.x = p1.y
    # b2 = boundbox2d(p2,p1)
    # b3 = boundbox3d()
    print(obj)
    # a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # b = np.array([0,0,0])
    # c = np.column_stack((a,b))
    # print(c)

    bbox3d = np.array([[0,0,0,1,1,1,2],[0,0,0,1,2,1,2]])
    a = bbox3d[:, None, 3:6]
    print(a)
    # b = template[None, :, :]
    # c = bbox3d[:, None, 3:6].tile(1, 8, 1)