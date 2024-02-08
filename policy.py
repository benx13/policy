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
with open("configs/benchmark7.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


logger = Logger()


################INIT_TRACKER############################
grab_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
grab_counter = Counter(grab_tracker, config['GRAB_TEMPRATURE'], 100)

transition_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
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

model_pose = YOLO('models/yolov8x-pose.mlpackage')


if config['BLUE']:
    model = coremltools.models.MLModel(config['MODEL_PATH_BLUE'])
else: 
    model = coremltools.models.MLModel(config['MODEL_PATH'])
################INIT_CAP############################
fvs = Fvs(path=config['VIDEO_FILE'])

frame_rate = fvs.stream.get(cv2.CAP_PROP_FPS)
skip_time = 0*60+0
if skip_time != 0:
    cap = cv2.VideoCapture(config['VIDEO_FILE'])
    skip_frames = int(frame_rate * skip_time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    fvs.stream = cap
fvs.start()
###################inti_flow###########################
prev_prev = cv2.cvtColor(fvs.read(), cv2.COLOR_BGR2GRAY)[config['FLOW_ZONE'][1]:config['FLOW_ZONE'][3], config['FLOW_ZONE'][0]:config['FLOW_ZONE'][2]]
prev = cv2.cvtColor(fvs.read(), cv2.COLOR_BGR2GRAY)[config['FLOW_ZONE'][1]:config['FLOW_ZONE'][3], config['FLOW_ZONE'][0]:config['FLOW_ZONE'][2]]
prev_diff = diffImg(prev_prev, prev, prev)
###################INIT_IMSHOW###########################
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the window title with the current coordinates of the mouse
        cv2.setWindowTitle('Window', f'Coordinates: ({x}, {y})')
cv2.namedWindow("Window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Window", 1920, 1080)
cv2.setMouseCallback('Window', mouse_callback)
########################################################
bags = 0
lenght = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT)/5)-10
#lenght = int(frame_rate * 22*60) - skip_frames
#print(f'len = {lenght}')

widthx, heightx = 1820, 1080
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 60, (widthx, heightx))


for _ in tqdm(range(lenght)):
    start = time.time()
    img = fvs.read()
    plot_time_on_frame(img, fvs.stream, frame_rate)
    print(img.shape)

    flow_img = img[config['FLOW_ZONE'][1]:config['FLOW_ZONE'][3], config['FLOW_ZONE'][0]:config['FLOW_ZONE'][2]]
    cv2.imshow('flow', flow_img)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2GRAY)
    diff = diffImg(prev_prev, prev, flow_img)
    flow = get_optica_flow(diff, prev_diff)
    prev_prev = prev
    prev = flow_img
    prev_diff = diff


    transition_counter.reset()
    grab_counter.reset()
    forward_counter.reset()
    backward_counter.reset()
    machine_counter.reset()

    resized_img = cv2.resize(img, (384, 224))

    reds = img[config['RED_ZONE'][1]:config['RED_ZONE'][3], config['RED_ZONE'][0]:config['RED_ZONE'][2]]
    reds2 = get_red(reds)

    if config['BLUE']:
        blues = get_blue(resized_img)
        input = preprocess_img(blues)
    else:
        input = preprocess_img(resized_img)
    mstart = time.time()
    results_pose = model_pose(img, imgsz=[224, 384], show=False)
    #print('rrrrrrrrr')
    try:
        for r in results_pose:
            #for j in r.keypoints:
                ##print(j)
                #for t in j.data:
            plot_keypoints_with_lines(img, r.keypoints.data[0].numpy())
        #print('rrrrrrrrr')
    except:
        pass
    print(input.size)
    results = model.predict({'image': input,
                             'iouThreshold': config['IOU_THRESHOLD'], 
                             'confidenceThreshold': config['CONFIDENCE_THRESHOLD']})
    mstop = time.time()

    if(np.sum(reds2[:,:,2])>0):
            transition_counter.update([1880, 100])
    
    for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
        if confidence > config['CONFIDENCE_THRESHOLD']:
            x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
            plot_rectangles1(img, x1,y1,x2,y2,confidence)
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),config['GRAB_ZONE']) or centroid_in_zone((x, y), (x1, y1, x2, y2),config['GRAB_ZONE_2'])):
                grab_counter.update([int(x), int(y)])
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),config['FORWARD_ZONE'])):
                forward_counter.update([int(x), int(y)])
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),config['BACKWARD_ZONE'])):
                backward_counter.update([int(x), int(y)])
            if(centroid_in_zone((x, y), (x1, y1, x2, y2),config['MACHINE_ZONE'])):
                machine_counter.update([int(x), int(y)])
    current_time = frame_to_hms(fvs.stream.get(cv2.CAP_PROP_POS_FRAMES), frame_rate)
    logger.update_flow(1 if flow > 50 else 0)
    #---------------
    flagXtransition = transition_counter.apply()
    if(flagXtransition):
        logger.update('transition', current_time)
    #---------------
    flagXgrab = grab_counter.apply()
    plot_path(img, grab_tracker.objects, grab_tracker, grab_history)
    if(flagXgrab):
        logger.update('grab', current_time)
    #---------------
    flagXforward = forward_counter.apply()
    plot_path(img, forward_tracker.objects, forward_tracker, forward_history)
    if(flagXforward):
        logger.update('forward', current_time)
    #---------------
    flagXbackward = backward_counter.apply()
    plot_path(img, backward_tracker.objects, backward_tracker, backward_history)
    if(flagXbackward):
        logger.update('backward', current_time)
    #---------------
    flagXmachine = machine_counter.apply()
    plot_path(img, machine_tracker.objects, machine_tracker, machine_history)
    if(flagXmachine):
        logger.update('machine', current_time)
    #---------------



    #overlay_region(img, config['GRAB_ZONE'], alpha=0.5)
    #overlay_region(img, config['GRAB_ZONE_2'], alpha=0.5)
    #overlay_region(img, config['FORWARD_ZONE'], alpha=0.5)
    #overlay_region(img, config['BACKWARD_ZONE'], alpha=0.5)
    #overlay_region(img, config['MACHINE_ZONE'], alpha=0.5)
    overlay_region(img, config['TOTAL_STATS_ZONE'], alpha=1)
    overlay_region(img, config['CURRENT_STATS_ZONE'], alpha=1)
    #------------------------
    plot_stats_ccurrent(img, config['CURRENT_STATS_ZONE'], logger.buffer, 'current_stats')
    plot_stats(img, config['TOTAL_STATS_ZONE'], logger.stats['total'], 'total_stats')
    #---------------------
    logger.update_logs()
    plot_logs(img, (0, 1060), logger.logs)
    #-----------------
    cv2.imshow('Window', img)
    #------------------
    stop = time.time()
    inference = mstop - mstart
    total = stop - start
    #------------------
    #print(f'total time:{total*1000} inference time: {inference*1000}')
    #print(logger.stats)
    #print(20*'-')
    #------------------
    Xsave = img[:, :1820,:]
    #print('-----',Xsave.shape)
    out.write(Xsave)
    if cv2.waitKey(1) == ord('q'):
        break
    
logger.update('transition', current_time)
fvs.stop()
out.release()
cv2.destroyAllWindows()
logger.save_results()




