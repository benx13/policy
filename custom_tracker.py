from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
 
class CentroidTracker():
	def __init__(self, maxDisappeared=50, minDistanece=150, direction=[]):
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
		###print(f'{direction_y} {direction_x}')
		return f'{direction_y} {direction_x}'

	def set_centroid(self, ID, centroid):
		self.objects[ID] = centroid
	def get_centroid(self):
		return self.objects[self.count]

	def update(self, new_centroid):
		self.flag = 0
		#print(self.disappeared)
		#print(self.objects)
		#print(self.count)
		if new_centroid == [] and self.disappeared:
			self.disappeared[self.count] += 1
			if self.disappeared[self.count] > self.maxDisappeared:
				self.deregister(self.count)
			return self.objects, self.flag
		if new_centroid:
			if len(self.objects) == 0:
				self.register(new_centroid)
			else:
				current_centroid = self.get_centroid()
				#print(f'centroid: {current_centroid} --> {new_centroid}')
				D = dist.euclidean(current_centroid, new_centroid)
				direction = self.get_direction(current_centroid, new_centroid)
				#print(f'direction ---> {direction}')

				if D < self.minDistanece:
					self.set_centroid(self.count, new_centroid)
					if len(self.direction) != 0 :
						if direction in self.direction:
							#print('here')
							self.flag = 1
							self.disappeared[self.count] = 0
					else:
						self.flag = 1
						self.disappeared[self.count] = 0
				#condition = D < self.minDistanece
				#if self.direction:
				#	condition = (D < self.minDistanece) and (direction in self.direction)
				#if condition:
				#	self.flag = 1
				#	self.disappeared[self.count] = 0
		#print(f'flag--->{self.flag}')
		return self.objects, self.flag

		