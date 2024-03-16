from scipy.spatial import distance as dist
from collections import OrderedDict
from circular_buffer import CircularBuffer
import numpy as np
from numba import njit
class Counter():
	def __init__(self, tracker, temprature, buffer_size=100):
		self.tracker = tracker
		self.tracking_history = CircularBuffer(buffer_size)
		self.id_dict = {}
		self.rects = []
		self.appeared_flag = 0
		self.flagX = 0
		self.temprature = temprature
		
	def reset(self):
		self.rects = []
		self.appeared_flag = 0
		self.flagX = 0
	
	def update(self, rect):
		self.rects = rect
		self.appeared_flag = 1

	def apply(self):
		transition_objects, self.flagX = self.tracker.update(self.rects)
		if(str(self.tracker.count) not in self.id_dict.keys()):
			self.id_dict[str(self.tracker.count)] = 0
		if(self.appeared_flag == 1 and self.id_dict[str(self.tracker.count)] == 0):
			self.tracking_history.append(self.flagX)
		else:
			self.tracking_history.append(0)
		if(self.flagX):
			if(self.id_dict[str(self.tracker.count)] == 0):
				if(self.tracking_history.sum() > self.temprature):
					self.id_dict[str(self.tracker.count)] = 1
					self.tracking_history.flush()
					return 1
		return 0