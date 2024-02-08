import cv2
import coremltools
from PIL import Image
import numpy as np
import time
from utils import *  
from custom_tracker import CentroidTracker
from collections import defaultdict
from circular_buffer import CircularBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt
from counter import Counter
from logger import Logger
from imutils.video import FileVideoStream as Fvs
import yaml
from ultralytics import YOLO
with open("configs/benchmarkcam.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


logger = Logger()


################INIT_TRACKER############################
grab_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
grab_counter = Counter(grab_tracker, config['GRAB_TEMPRATURE'], 100)

transition_tracker = CentroidTracker(maxDisappeared=2000, minDistanece=200)
transition_counter = Counter(transition_tracker, config['TRANSITION_TEMPRATURE'], 100)

forward_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200, direction=config['FORWARD_DIRECTIONS'])
forward_counter = Counter(forward_tracker, config['FORWARD_TEMPRATURE'], 100)

backward_tracker = CentroidTracker(maxDisappeared=2000, minDistanece=250, direction=config['BACKWARD_DIRECTIONS'])
backward_counter = Counter(backward_tracker, config['BACKWARD_TEMPRATURE'], 100)

machine_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)#, direction=config['MACHINE_DIRECTIONS'])
machine_counter = Counter(machine_tracker, config['MACHINE_TEMPRATURE'], 100)


grab_history = defaultdict(lambda: [])
transition_history = defaultdict(lambda: [])
forward_history = defaultdict(lambda: [])
backward_history = defaultdict(lambda: [])
machine_history = defaultdict(lambda: [])



if config['BLUE']:
    model = coremltools.models.MLModel(config['MODEL_PATH_BLUE'])
else: 
    model = coremltools.models.MLModel(config['MODEL_PATH'])
################INIT_CAP############################
cap = cv2.VideoCapture('videos/benchmark_cam_02.mov')
frame_rate = cap.get(cv2.CAP_PROP_FPS)
skip_time = 0*60+0
if skip_time != 0:
    skip_frames = int(frame_rate * skip_time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)


print(cap.read()[1].shape)

###################inti_flow###########################
prev_prev = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)[config['FLOW_ZONE'][1]:config['FLOW_ZONE'][3], config['FLOW_ZONE'][0]:config['FLOW_ZONE'][2]]
prev = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)[config['FLOW_ZONE'][1]:config['FLOW_ZONE'][3], config['FLOW_ZONE'][0]:config['FLOW_ZONE'][2]]
prev_diff = diffImg(prev_prev, prev, prev)
###################INIT_IMSHOW###########################
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the window title with the current coordinates of the mouse
        cv2.setWindowTitle('Window', f'Coordinates: ({x}, {y})')
cv2.namedWindow("Window", cv2.WINDOW_NORMAL) 
#cv2.resizeWindow("Window", 1920, 1080)
cv2.setMouseCallback('Window', mouse_callback)
########################################################
bags = 0
#lenght = int(frame_rate * 22*60) - skip_frames
#print(f'len = {lenght}')

TRANSITION_ZONE_01 = [464, 303, 473, 309]
TRANSITION_ZONE_02 = [478, 304, 487, 308]
TRANSITION_ZONE_03 = [478, 355, 487, 361]
TRANSITION_ZONE_04 = [542, 355, 552, 362]
TRANSITION_ZONE_05 = [475, 373, 484, 380]
TRANSITION_ZONE_06 = [539, 373, 548, 380]
TRANSITION_ZONE_07 = [582, 371, 592, 376]
TRANSITION_ZONE_08 = [470, 388, 480, 396]
TRANSITION_ZONE_09 = [532, 390, 543, 399]
TRANSITION_ZONE_10 = [580, 389, 590, 395]

lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/3.9  )-10

print(f'len = {lenght}')
for _ in tqdm(range(lenght)):
    start = time.time()
    flag, img = cap.read()
    #cv2.imshow('img', img)
    #plot_time_on_frame(img, cap, frame_rate)
   

    z1 = binary_image(get_orange(img[TRANSITION_ZONE_01[1]:TRANSITION_ZONE_01[3], TRANSITION_ZONE_01[0]:TRANSITION_ZONE_01[2]]))
    z2 = binary_image(get_orange(img[TRANSITION_ZONE_02[1]:TRANSITION_ZONE_02[3], TRANSITION_ZONE_02[0]:TRANSITION_ZONE_02[2]]))
    z3 = binary_image(get_orange(img[TRANSITION_ZONE_03[1]:TRANSITION_ZONE_03[3], TRANSITION_ZONE_03[0]:TRANSITION_ZONE_03[2]]))
    z4 = binary_image(get_orange(img[TRANSITION_ZONE_04[1]:TRANSITION_ZONE_04[3], TRANSITION_ZONE_04[0]:TRANSITION_ZONE_04[2]]))
    z5 = binary_image(get_orange(img[TRANSITION_ZONE_05[1]:TRANSITION_ZONE_05[3], TRANSITION_ZONE_05[0]:TRANSITION_ZONE_05[2]]))
    z6 = binary_image(get_orange(img[TRANSITION_ZONE_06[1]:TRANSITION_ZONE_06[3], TRANSITION_ZONE_06[0]:TRANSITION_ZONE_06[2]]))
    z7 = binary_image(get_orange(img[TRANSITION_ZONE_07[1]:TRANSITION_ZONE_07[3], TRANSITION_ZONE_07[0]:TRANSITION_ZONE_07[2]]))
    z8 = binary_image(get_orange(img[TRANSITION_ZONE_08[1]:TRANSITION_ZONE_08[3], TRANSITION_ZONE_08[0]:TRANSITION_ZONE_08[2]]))
    z9 = binary_image(get_orange(img[TRANSITION_ZONE_09[1]:TRANSITION_ZONE_09[3], TRANSITION_ZONE_09[0]:TRANSITION_ZONE_09[2]]))
    z10 = binary_image(get_orange(img[TRANSITION_ZONE_10[1]:TRANSITION_ZONE_10[3], TRANSITION_ZONE_10[0]:TRANSITION_ZONE_10[2]]))
 
    transition_counter.reset()
    grab_counter.reset()
    forward_counter.reset()
    backward_counter.reset()
    machine_counter.reset()

    resized_img = cv2.resize(img, (640, 640))

    #reds = img[config['RED_ZONE'][1]:config['RED_ZONE'][3], config['RED_ZONE'][0]:config['RED_ZONE'][2]]
    #reds2 = get_red(reds)


    if(np.sum(z1)>255*7 and 
       np.sum(z2)>255*7 and 
       np.sum(z3)>255*7 and 
       np.sum(z4)>255*7 and 
       np.sum(z5)>255*7 and
       np.sum(z6)>255*7 and
       np.sum(z7)>255*7 and
       np.sum(z8)>255*7 and
       np.sum(z9)>255*7 and
       np.sum(z10)>255*7 
       ):
            transition_counter.update([1880, 100])
    

    current_time = frame_to_hms(cap.get(cv2.CAP_PROP_POS_FRAMES), frame_rate)
    #---------------
    flagXtransition = transition_counter.apply()
    #print(transition_counter.tracking_history)
    if(flagXtransition):
        logger.update('transition', current_time)
   




    #overlay_region(img, TRANSITION_ZONE_01, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_02, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_03, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_04, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_05, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_06, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_07, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_08, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_09, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_10, alpha=0.5)





    #---------------------
    logger.update_logs()
    #plot_logs(img, (0, 700), logger.logs)
    #cv2.imshow('Window', img)

    #------------------
    #------------------
    #------------------
    #key = cv2.waitKey(1)

    #if key == ord('q'):
    #    break
    #elif key == ord('p'):
    #    paused = not paused  # Toggle the pause state

logger.update('transition', current_time)
cap.release()
cv2.destroyAllWindows()
logger.save_results()




