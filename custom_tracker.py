from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 
class CentroidTracker():
	def __init__(self, maxDisappeared=50, minDistanece=150, direction=None):
		self.objects = {}
		self.disappeared = {}
		self.nextObjectID = 1
		self.count = 0
		self.flag = 0
		self.maxDisappeared = maxDisappeared
		self.minDistanece = minDistanece
		self.direction = direction

	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1
		self.count += 1
		
	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
		
	def get_direction(self, currentCentroid, newCentroid):
		x1, y1 = currentCentroid
		x2, y2 = newCentroid
		direction_x = ""
		direction_y = ""

		if x2 > x1:
			direction_x = "right"
		elif x2 < x1:
			direction_x = "left"
		if y2 > y1:
			direction_y = "down"
		elif y2 < y1:
			direction_y = "up"
		#print(f'{direction_y} {direction_x}')
		return f'{direction_y} {direction_x}'


def update(self, centroid):
    if centroid == None:
        self.disappeared[self.count] += 1
        if self.disappeared[self.count] > self.maxDisappeared:
            self.deregister(self.count)
        return self.objects, self.flag

    if len(self.objects) == 0:
        self.register(centroid)