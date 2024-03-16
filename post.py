
from utils import *  
from custom_tracker import CentroidTracker
from counter import Counter
from logger import Logger
import yaml
import numpy as np

class Post():
    def __init__(self, config) -> None:
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.out_dir = self.config['OUTDIR']
        self.name = self.config['NAME']
        self.logger = Logger()

        self.grab_tracker = CentroidTracker(maxDisappeared=400, minDistanece=200)
        self.grab_counter = Counter(self.grab_tracker, self.config['GRAB_TEMPRATURE'], 100)

        self.transition_tracker = CentroidTracker(maxDisappeared=2000, minDistanece=200)
        self.transition_counter = Counter(self.transition_tracker, self.config['TRANSITION_TEMPRATURE'], self.config['TRANSITION_MAX'])
        
        self.prev_tran = "00:00:00"
        self.forward_tracker = CentroidTracker(maxDisappeared=800, minDistanece=200, direction=self.config['FORWARD_DIRECTIONS'])
        self.forward_counter = Counter(self.forward_tracker, self.config['FORWARD_TEMPRATURE'], 100)

        self.backward_tracker = CentroidTracker(maxDisappeared=800, minDistanece=250, direction=self.config['BACKWARD_DIRECTIONS'])
        self.backward_counter = Counter(self.backward_tracker, self.config['BACKWARD_TEMPRATURE'], 100)

        self.machine_tracker = CentroidTracker(maxDisappeared=400, minDistanece=200)#, direction=self.config['MACHINE_DIRECTIONS'])
        self.machine_counter = Counter(self.machine_tracker, self.config['MACHINE_TEMPRATURE'], 100)

    def update_objects(self, objects):
        x, y, x1, y1, x2, y2 = objects
        if(centroid_in_zone((x, y), (x1, y1, x2, y2),self.config['GRAB_ZONE'])):# or centroid_in_zone((x, y), (x1, y1, x2, y2),self.config['GRAB_ZONE_2'])):
            self.grab_counter.update([int(x), int(y)])
        if(centroid_in_zone((x, y), (x1, y1, x2, y2),self.config['FORWARD_ZONE'])):
            self.forward_counter.update([int(x), int(y)])
        if(centroid_in_zone((x, y), (x1, y1, x2, y2),self.config['BACKWARD_ZONE'])):
            self.backward_counter.update([int(x), int(y)])
        if(centroid_in_zone((x, y), (x1, y1, x2, y2),self.config['MACHINE_ZONE'])):
            self.machine_counter.update([int(x), int(y)])
    
    def forward_transition(self, current_time):
        flagXtransition = self.transition_counter.apply()
        if(flagXtransition):
            trt = hms_difference(current_time, self.prev_tran)
            if (int(trt[3])*360 + int(trt[4])*60) > 150:
                self.prev_tran = current_time
                print(f'transition {self.name} -->{current_time}')
                self.logger.update('transition', current_time)
                self.logger.stats['t2t_list'].append(trt)
                self.logger.buffer['t2t'] = trt
                self.logger.save_results(self.out_dir)

    def forward(self, current_time):
        flagXtransition = self.transition_counter.apply()
        if(flagXtransition):
            trt = hms_difference(current_time, self.prev_tran)
            if (int(trt[3])*360 + int(trt[4])*60) > 150:
                self.prev_tran = current_time
                print(f'transition {self.name} -->{current_time}')
                self.logger.update('transition', current_time)
                self.logger.stats['t2t_list'].append(trt)
                self.logger.buffer['t2t'] = trt
                self.logger.save_results(self.out_dir)

        flagXgrab = self.grab_counter.apply()
        if(flagXgrab):
            self.logger.update('grab', current_time)

        flagXforward = self.forward_counter.apply()
        if(flagXforward):
            self.logger.update('forward', current_time)

        flagXbackward = self.backward_counter.apply()
        if(flagXbackward):
            self.logger.update('backward', current_time)

        flagXmachine = self.machine_counter.apply()
        if(flagXmachine):
            self.logger.update('machine', current_time)

    def reset(self):

        self.transition_counter.reset()
        self.grab_counter.reset()
        self.forward_counter.reset()
        self.backward_counter.reset()
        self.machine_counter.reset()

    def update_transition(self, img):
        z1 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_01'][1]:self.config['TRANSITION_ZONE_01'][3], self.config['TRANSITION_ZONE_01'][0]:self.config['TRANSITION_ZONE_01'][2]]))
        z2 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_02'][1]:self.config['TRANSITION_ZONE_02'][3], self.config['TRANSITION_ZONE_02'][0]:self.config['TRANSITION_ZONE_02'][2]]))
        z3 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_03'][1]:self.config['TRANSITION_ZONE_03'][3], self.config['TRANSITION_ZONE_03'][0]:self.config['TRANSITION_ZONE_03'][2]]))
        z4 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_04'][1]:self.config['TRANSITION_ZONE_04'][3], self.config['TRANSITION_ZONE_04'][0]:self.config['TRANSITION_ZONE_04'][2]]))
        z5 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_05'][1]:self.config['TRANSITION_ZONE_05'][3], self.config['TRANSITION_ZONE_05'][0]:self.config['TRANSITION_ZONE_05'][2]]))
        z6 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_06'][1]:self.config['TRANSITION_ZONE_06'][3], self.config['TRANSITION_ZONE_06'][0]:self.config['TRANSITION_ZONE_06'][2]]))
        z7 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_07'][1]:self.config['TRANSITION_ZONE_07'][3], self.config['TRANSITION_ZONE_07'][0]:self.config['TRANSITION_ZONE_07'][2]]))
        z8 = binary_image(get_orange(img[self.config['TRANSITION_ZONE_08'][1]:self.config['TRANSITION_ZONE_08'][3], self.config['TRANSITION_ZONE_08'][0]:self.config['TRANSITION_ZONE_08'][2]]))

        if( np.sum(z1) > 255*self.config['TRANSITION_TRESH'][0] and 
            np.sum(z2) > 255*self.config['TRANSITION_TRESH'][1] and
            ( 
            (1 if np.sum(z3) > 255*self.config['TRANSITION_TRESH'][2] else 0) + 
            (1 if np.sum(z4) > 255*self.config['TRANSITION_TRESH'][3] else 0) + 
            (1 if np.sum(z5) > 255*self.config['TRANSITION_TRESH'][4] else 0) + 
            (1 if np.sum(z6) > 255*self.config['TRANSITION_TRESH'][5] else 0) + 
            (1 if np.sum(z7) > 255*self.config['TRANSITION_TRESH'][6] else 0) + 
            (1 if np.sum(z8) > 255*self.config['TRANSITION_TRESH'][7] else 0) ) > self.config['MIN_TRAN']
            ):
                self.transition_counter.update([1880, 100])
        
    def close(self, current_time):
        self.logger.update('transition', current_time)
        self.logger.save_results(self.out_dir)