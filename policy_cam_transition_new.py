import cv2
import numpy as np
import time
from utils import *  
from custom_tracker import CentroidTracker
from collections import defaultdict
from tqdm import tqdm
from counter import Counter
from logger import Logger
import yaml
with open("configs/benchmarkcam.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


logger = Logger()


transition_tracker = CentroidTracker(maxDisappeared=2000, minDistanece=200)
transition_counter = Counter(transition_tracker, 50, 200)




#cap = cv2.VideoCapture("rtsp://benx:benxbenx13@192.168.0.101:554/videoMain")
cap = cv2.VideoCapture("videos/benchmark_cam_14.mov")
frame_rate = cap.get(cv2.CAP_PROP_FPS)
#frame_rate = 25
skip_time = 0*3600 + 0*60+0
if skip_time != 0:
    skip_frames = int(frame_rate * skip_time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)


print(cap.read()[1].shape)
print(frame_rate)
###################INIT_IMSHOW###########################
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the window title with the current coordinates of the mouse
        tmp = cv2.cvtColor(img[TRANSITION_ZONE_01[1]:TRANSITION_ZONE_01[3], TRANSITION_ZONE_01[0]:TRANSITION_ZONE_01[2]], cv2.COLOR_BGR2HSV)
        color = tmp[y][x]
        cv2.setWindowTitle('z4', f'Coordinates: ({x}, {y}), colorHSV: ({color})')
cv2.namedWindow("z4", cv2.WINDOW_NORMAL) 
#cv2.resizeWindow("Window", 1920, 1080)
cv2.setMouseCallback('z4', mouse_callback)
########################################################
bags = 0
#lenght = int(frame_rate * 22*60) - skip_frames
#print(f'len = {lenght}')

TRANSITION_ZONE_01= [487, 298, 527, 317]
TRANSITION_ZONE_02= [464, 324, 486, 342]
TRANSITION_ZONE_03= [489, 324, 512, 343]
TRANSITION_ZONE_04= [522, 326, 550, 345]
TRANSITION_ZONE_05= [557, 328, 582, 348]
TRANSITION_ZONE_06= [479, 344, 507, 365]
TRANSITION_ZONE_07= [519, 346, 544, 367]
TRANSITION_ZONE_08= [552, 349, 577, 369]

lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/1  )-10
paused = False
#print(f'len = {lenght}')
prev = '00:00:00'
for i in tqdm(range(lenght)):
#while True:

    flag, img = cap.read()
    #cv2.imshow('img', img)
    plot_time_on_frame(img, cap, frame_rate)
   
    transition_counter.reset()

    z1 = binary_image(get_orange(img[TRANSITION_ZONE_01[1]:TRANSITION_ZONE_01[3], TRANSITION_ZONE_01[0]:TRANSITION_ZONE_01[2]]))
    z2 = binary_image(get_orange(img[TRANSITION_ZONE_02[1]:TRANSITION_ZONE_02[3], TRANSITION_ZONE_02[0]:TRANSITION_ZONE_02[2]]))
    z3 = binary_image(get_orange(img[TRANSITION_ZONE_03[1]:TRANSITION_ZONE_03[3], TRANSITION_ZONE_03[0]:TRANSITION_ZONE_03[2]]))
    z4 = binary_image(get_orange(img[TRANSITION_ZONE_04[1]:TRANSITION_ZONE_04[3], TRANSITION_ZONE_04[0]:TRANSITION_ZONE_04[2]]))
    z5 = binary_image(get_orange(img[TRANSITION_ZONE_05[1]:TRANSITION_ZONE_05[3], TRANSITION_ZONE_05[0]:TRANSITION_ZONE_05[2]]))
    z6 = binary_image(get_orange(img[TRANSITION_ZONE_06[1]:TRANSITION_ZONE_06[3], TRANSITION_ZONE_06[0]:TRANSITION_ZONE_06[2]]))
    z7 = binary_image(get_orange(img[TRANSITION_ZONE_07[1]:TRANSITION_ZONE_07[3], TRANSITION_ZONE_07[0]:TRANSITION_ZONE_07[2]]))
    z8 = binary_image(get_orange(img[TRANSITION_ZONE_08[1]:TRANSITION_ZONE_08[3], TRANSITION_ZONE_08[0]:TRANSITION_ZONE_08[2]]))


    if(np.sum(z1)>255*20 and 
       np.sum(z2)>255*20 and 
       (
       (1 if np.sum(z3)>255*20 else 0) + 
       (1 if np.sum(z4)>255*20 else 0) + 
       (1 if np.sum(z5)>255*20 else 0) + 
       (1 if np.sum(z6)>255*20 else 0) + 
       (1 if np.sum(z7)>255*20 else 0) + 
       (1 if np.sum(z8)>255*20 else 0) ) > 4

       ):

            transition_counter.update([1880, 100])
    

    current_time = frame_to_hms(cap.get(cv2.CAP_PROP_POS_FRAMES), frame_rate)
    #---------------252


    flagXtransition = transition_counter.apply()
    #print(transition_counter.tracking_history)

    if(flagXtransition):
        trt = hms_difference(current_time, prev)
        print(int(trt[3])*360 + int(trt[4])*60)
        if (int(trt[3])*360 + int(trt[4])*60) > 150:
            prev = current_time
            print(f'transition -->{current_time}, trt -->{trt}')
            logger.update('transition', current_time)
    logger.update_logs()
   




    '''










    cv2.imshow('z4', (img[TRANSITION_ZONE_02[1]:TRANSITION_ZONE_02[3], TRANSITION_ZONE_02[0]:TRANSITION_ZONE_02[2]]))
    cv2.imshow('z4_y', get_orange(img[TRANSITION_ZONE_02[1]:TRANSITION_ZONE_02[3], TRANSITION_ZONE_02[0]:TRANSITION_ZONE_02[2]]))

    print(
         (
       (1 if np.sum(z3)>255*20 else 0) + 
       (1 if np.sum(z4)>255*20 else 0) + 
       (1 if np.sum(z5)>255*20 else 0) + 
       (1 if np.sum(z6)>255*20 else 0) + 
       (1 if np.sum(z7)>255*20 else 0) + 
       (1 if np.sum(z8)>255*20 else 0) ) 
         
    )
    print(np.sum(z1)>255*20) 
    print(np.sum(z2)>255*20) 
    #print(f"flagXtransition --> {flagXtransition}")
    #print(f"trt --> {trt}")
    print('--------')

    if(np.sum(z1)>255*20 and 
       np.sum(z2)>255*20 and 
       (
       (1 if np.sum(z3)>255*20 else 0) + 
       (1 if np.sum(z4)>255*20 else 0) + 
       (1 if np.sum(z5)>255*20 else 0) + 
       (1 if np.sum(z6)>255*20 else 0) + 
       (1 if np.sum(z7)>255*20 else 0) + 
       (1 if np.sum(z8)>255*20 else 0) ) > 3

       #np.sum(z8)>255*1 and
       #np.sum(z9)>255*1 and
       #np.sum(z10)>255*1
       ):
            overlay_region(img, TRANSITION_ZONE_01, alpha=1)
            overlay_region(img, TRANSITION_ZONE_02, alpha=1)
            overlay_region(img, TRANSITION_ZONE_03, alpha=1)
            overlay_region(img, TRANSITION_ZONE_04, alpha=1)
            overlay_region(img, TRANSITION_ZONE_05, alpha=1)
            overlay_region(img, TRANSITION_ZONE_06, alpha=1)
            overlay_region(img, TRANSITION_ZONE_07, alpha=1)
            overlay_region(img, TRANSITION_ZONE_08, alpha=1)
            #transition_counter.update([1880, 100])
    
    overlay_region(img, TRANSITION_ZONE_01, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_02, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_03, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_04, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_05, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_06, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_07, alpha=0.5)
    overlay_region(img, TRANSITION_ZONE_08, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_09, alpha=0.5)
    #overlay_region(img, TRANSITION_ZONE_10, alpha=0.5)


    plot_logs(img, (0, 700), logger.logs)
    cv2.imshow('Window', img)

    key = cv2.waitKey(30)

    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused  # Toggle the pause state
    '''

logger.update('transition', current_time)
cap.release()
cv2.destroyAllWindows()
logger.save_results('transitions_benchmark_cam_04')




