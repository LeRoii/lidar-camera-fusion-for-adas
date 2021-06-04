import numpy as np
from objectbase import objectbase
from filterpy.kalman import KalmanFilter
from numba import jit
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
import attr

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

# @attr.s
class trackedobj(objectbase):
    count = 0
    def __init__(self, obj):
        objectbase.__init__(self, obj.objtype, obj.bbox3d, obj.bbox2d, obj.confidence)
        self.age = 1
        self.lostCnt = 0
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.id = trackedobj.count
        trackedobj.count += 1
        print('ctor')
    
    def update(self,obj):
        self.bbox3d = obj.bbox3d
        self.bbox2d = obj.bbox2d
        self.confidence = obj.confidence

    # age = attr.ib(type=int, default=0)
    # kf = attr.ib(type=KalmanFilter, default=KalmanFilter(dim_x=10, dim_z=7))

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

        self.objDistArray = np.zeros((len(objs), len(self.trackerList)), dtype=np.float32)
        for i, detobj in enumerate(objs):
            for j, tracked_obj in enumerate(self.trackerList):
                self.objDistArray[i,j] = self.calcDist(detobj, tracked_obj)

        # hougarian algorithm
        row_ind, col_ind = linear_sum_assignment(-self.objDistArray)
        matched_indices = np.stack((row_ind, col_ind), axis=1)

        # update matched trackers
        deleteMatchIdx = []
        for i, pair in enumerate(matched_indices):
            if self.trackerList[pair[1]].objtype != objs[pair[0]].objtype:
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
        
        return


# if __name__=='__main__':
#     obj = objectbase()
#     tobj = trackedobj(obj)
#     print(tobj)
#     print(tobj.bbox3d)