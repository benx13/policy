from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 
class CentroidTracker():
	def __init__(self, maxDisappeared=50, minDistanece=100, direction=None):
		self.nextObjectID = 1
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.count = 0
		self.maxDisappeared = maxDisappeared
		self.minDistanece = minDistanece
		self.direction = direction
		self.flag = 0

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
		return f'{direction_y} {direction_x}'
	
	def update(self, rects):
		self.flag = 0
		if len(rects) == 0:
			for objectID in self.disappeared.copy().keys():
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				objectID = objectIDs[row]
				direction = self.get_direction(objectCentroids[0], inputCentroids[0])

				condition = D[row, col] < self.minDistanece
				print(direction)
				if self.direction:
					condition &= (direction in self.direction )
				if condition:
						self.flag = 1
						self.objects[objectID] = inputCentroids[col]
						self.disappeared[objectID] = 0
				usedRows.add(row)
				usedCols.add(col)
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		return self.objects