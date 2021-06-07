import numpy as np
from objectbase import objectbase, point3d
from filterpy.kalman import KalmanFilter
from numba import jit
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
import attr
import torch

import matplotlib.pyplot as plt

@jit          
def poly_area(x,y):
	""" Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit         
def box3d_vol(corners):
	''' corners: (8,3) no assumption on axis direction '''
	a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
	b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
	c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
	return a*b*c

@jit          
def convex_hull_intersection(p1, p2):
	""" Compute area of two convex hull's intersection area.
		p1,p2 are a list of (x,y) tuples of hull vertices.
		return a list of (x,y) for the intersection and its volume
	"""
	inter_p = polygon_clip(p1,p2)
	if inter_p is not None:
		hull_inter = ConvexHull(inter_p)
		return inter_p, hull_inter.volume
	else:
		return None, 0.0  

def polygon_clip(subjectPolygon, clipPolygon):
	""" Clip a polygon with another polygon.
	Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

	Args:
		subjectPolygon: a list of (x,y) 2d points, any polygon.
		clipPolygon: a list of (x,y) 2d points, has to be *convex*
	Note:
		**points have to be counter-clockwise ordered**

	Return:
		a list of (x,y) vertex point for the intersection polygon.
	"""
	def inside(p):
		return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
 
	def computeIntersection():
		dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
		dp = [s[0] - e[0], s[1] - e[1]]
		n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
		n2 = s[0] * e[1] - s[1] * e[0] 
		n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
		return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]
 
	outputList = subjectPolygon
	cp1 = clipPolygon[-1]
 
	for clipVertex in clipPolygon:
		cp2 = clipVertex
		inputList = outputList
		outputList = []
		s = inputList[-1]
 
		for subjectVertex in inputList:
			e = subjectVertex
			if inside(e):
				if not inside(s): outputList.append(computeIntersection())
				outputList.append(e)
			elif inside(s): outputList.append(computeIntersection())
			s = e
		cp1 = cp2
		if len(outputList) == 0: return None
	return (outputList)

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
    a = boxes3d[:, None, 3:6]
    b = template[None, :, :]
    c = boxes3d[:, None, 3:6].repeat(1, 8, 1)
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

# @attr.s
class trackedobj(objectbase):
    count = 0
    def __init__(self, obj):
        objectbase.__init__(self, obj.objtype, obj.bbox3d, obj.bbox2d, obj.confidence)
        self.age = 1
        self.lostCnt = 0
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function
                                [0,1,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0]])
        self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.
        self.kf.x[:7] = np.array([obj.bbox3d.centerpoint.x,obj.bbox3d.centerpoint.y,obj.bbox3d.centerpoint.z,
        obj.bbox3d.length, obj.bbox3d.width, obj.bbox3d.height, obj.bbox3d.theta]).reshape((7,1))

		# self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.R[6,6] = 10
        self.id = trackedobj.count
        trackedobj.count += 1

        self.measures = np.empty(shape=(1,7))
        self.filterrets = np.empty(shape=(1,7))

    def restoreState(self):
        bbox = self.kf.x[:7].reshape((1,7))
        bbox_corners3d = boxes_to_corners_3d(bbox)
        corners = bbox_corners3d.reshape((bbox_corners3d.shape[0],-1))

        self.bbox3d.centerpoint.x = self.kf.x[0][0]
        self.bbox3d.centerpoint.y = self.kf.x[1][0]
        self.bbox3d.centerpoint.z = self.kf.x[2][0]
        self.bbox3d.theta = self.kf.x[6][0]
        self.bbox3d.length = self.kf.x[3][0]
        self.bbox3d.width = self.kf.x[4][0]
        self.bbox3d.height = self.kf.x[5][0]
        for i in range(8):
            self.bbox3d.corners[i] = point3d(corners[0,i*3],corners[0,1+i*3],corners[0,2+i*3])
    
    def update(self,obj):
        # if obj.bbox3d.theta >= np.pi * 2: obj.bbox3d.theta -= np.pi * 2
        if obj.bbox3d.theta < 0: obj.bbox3d.theta += np.pi * 2
        if abs(obj.bbox3d.theta - self.kf.x[6][0]) > np.pi / 2: 
             self.kf.x[6][0] = obj.bbox3d.theta

        mesured = np.array([obj.bbox3d.centerpoint.x,obj.bbox3d.centerpoint.y,obj.bbox3d.centerpoint.z,
        obj.bbox3d.length, obj.bbox3d.width, obj.bbox3d.height, obj.bbox3d.theta]).reshape((7,1))

        state = [[obj.bbox3d.centerpoint.x,obj.bbox3d.centerpoint.y,obj.bbox3d.centerpoint.z,
        obj.bbox3d.length, obj.bbox3d.width, obj.bbox3d.height, obj.bbox3d.theta]]
        self.measures = np.append(self.measures,state,axis=0)

        self.kf.update(mesured)
        self.restoreState()

        # self.bbox3d = obj.bbox3d
        self.bbox2d = obj.bbox2d
        self.confidence = obj.confidence

        state = [[self.bbox3d.centerpoint.x,self.bbox3d.centerpoint.y,self.bbox3d.centerpoint.z,
        self.bbox3d.length, self.bbox3d.width, self.bbox3d.height, self.bbox3d.theta]]
        self.filterrets = np.append(self.filterrets,state,axis=0)

        # debug kf
        # if self.id == 32 and self.filterrets.shape[0] > 40:
        #     print('asda')
        #     fig = plt.figure()
        #     plt.plot(range(self.filterrets.shape[0]), self.filterrets[:,6],  color='lightblue', linewidth=3)
        #     plt.plot(range(self.filterrets.shape[0]), self.measures[:,6],  color='red', linewidth=3)
        #     plt.show()

    def predict(self):
        self.kf.predict()

        self.restoreState()

class tracker():
    def __init__(self):
        self.trackerList = []
        self.objDistArray = np.empty([])

    def calcDist(self, deted, tracked):
        rect1 = [(deted.bbox3d.corners[i].x, deted.bbox3d.corners[i].y) for i in range(3,-1,-1)]
        rect2 = [(tracked.bbox3d.corners[i].x, tracked.bbox3d.corners[i].y) for i in range(3,-1,-1)]
        area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
        area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

        _, inter_area = convex_hull_intersection(rect1, rect2)

        iou_2d = inter_area/(area1+area2-inter_area)
        ymax = min(deted.bbox3d.corners[4].z, tracked.bbox3d.corners[4].z)
        ymin = max(deted.bbox3d.corners[0].z, tracked.bbox3d.corners[0].z)
        inter_vol = inter_area * max(0.0, ymax-ymin)
        vol1 = deted.bbox3d.length * deted.bbox3d.width * deted.bbox3d.height
        vol2 = tracked.bbox3d.length * tracked.bbox3d.width * tracked.bbox3d.height
        iou = inter_vol / (vol1 + vol2 - inter_vol)
        return iou

    def update(self, objs):
        def removeTarget(obj):
            if obj.age < 5:
                if obj.lostCnt > 1:
                    return True
            else:
                if obj.lostCnt > 3:
                    return True
            return False

        if len(self.trackerList) == 0:
            for obj in objs:
                self.trackerList.append(trackedobj(obj))
            return
        print('tracker size:{}, det size:{}'.format(len(self.trackerList),len(objs)))
        # filter predict
        for tracked in self.trackerList:
            tracked.predict()

        self.objDistArray = np.zeros((len(objs), len(self.trackerList)), dtype=np.float32)
        for i, detobj in enumerate(objs):
            for j, tracked_obj in enumerate(self.trackerList):
                self.objDistArray[i,j] = self.calcDist(detobj, tracked_obj)

        # match
        row_ind, col_ind = linear_sum_assignment(-self.objDistArray)
        matched_indices = np.stack((row_ind, col_ind), axis=1)

        # update matched trackers
        deleteMatchIdx = []
        for i, pair in enumerate(matched_indices):
            if self.trackerList[pair[1]].objtype != objs[pair[0]].objtype or self.objDistArray[pair[0],pair[1]] == .0:
                deleteMatchIdx.append(i)
            else:
                self.trackerList[pair[1]].update(objs[pair[0]])
        matched_indices = np.delete(matched_indices, deleteMatchIdx, 0)

        unmatched_detections = []
        for d, det in enumerate(objs):
            if (d not in matched_indices[:, 0]): 
                unmatched_detections.append(d)
                self.trackerList.append(trackedobj(det))
        unmatched_trackers = []
        for t, trk in enumerate(self.trackerList):
            if (t not in matched_indices[:, 1]):
                trk.lostCnt += 1
                unmatched_trackers.append(t)
            else:
                trk.age += 1
                trk.lostCnt = 0

            if removeTarget(trk):
                self.trackerList.pop(t)
        
        print('unmatched tracker:{}, unmatched det :{}'.format(unmatched_trackers, unmatched_detections))
        
        return


# if __name__=='__main__':
#     obj = objectbase()
#     tobj = trackedobj(obj)
#     print(tobj)
#     print(tobj.bbox3d)