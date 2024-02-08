import cv2
import coremltools
import numpy as np
import time
from utils import *  
from custom_tracker import CentroidTracker
from tqdm import tqdm
from counter import Counter
from logger import Logger
from imutils.video import FileVideoStream as Fvs
import yaml
with open("configs/benchmark8.yaml") as f:
      config = yaml.load(f, Loader=yaml.FullLoader)


logger = Logger()


################INIT_TRACKER############################
grab_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
grab_counter = Counter(grab_tracker, config['GRAB_TEMPRATURE'], 100)

transition_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)
transition_counter = Counter(transition_tracker, config['TRANSITION_TEMPRATURE'], 100)

forward_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200, direction=config['FORWARD_DIRECTIONS'])
forward_counter = Counter(forward_tracker, config['FORWARD_TEMPRATURE'], 100)

backward_tracker = CentroidTracker(maxDisappeared=2000, minDistanece=300)#, direction=config['BACKWARD_DIRECTIONS'])
backward_counter = Counter(backward_tracker, config['BACKWARD_TEMPRATURE'], 100)

machine_tracker = CentroidTracker(maxDisappeared=750, minDistanece=200)#, direction=config['MACHINE_DIRECTIONS'])
machine_counter = Counter(machine_tracker, config['MACHINE_TEMPRATURE'], 100)


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
########################################################
lenght = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT)/1  )-10

print(f'len = {lenght}')
for _ in tqdm(range(lenght)):
    start = time.time()
    img = fvs.read()
    plot_time_on_frame(img, fvs.stream, frame_rate)
    stop=time.time()

    flow_img = img[config['FLOW_ZONE'][1]:config['FLOW_ZONE'][3], config['FLOW_ZONE'][0]:config['FLOW_ZONE'][2]]
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
        blues = getXwhite(resized_img)
        input = preprocess_img(blues)
    else:
        input = preprocess_img(resized_img)

    results = model.predict({'image': input, 
                             'iouThreshold': config['IOU_THRESHOLD'], 
                             'confidenceThreshold': config['CONFIDENCE_THRESHOLD']})

    if(np.sum(reds2[:,:,2])>0):
            transition_counter.update([1880, 100])
    
    for confidence, (xn, yn, widthn, heightn) in zip(results['confidence'], results['coordinates']):
        if confidence > config['CONFIDENCE_THRESHOLD']:
            x, y, x1, y1, x2, y2 = get_coordinates(img, xn, yn, widthn, heightn)
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
    if(flagXgrab):
        logger.update('grab', current_time)
    #---------------
    flagXforward = forward_counter.apply()
    if(flagXforward):
        logger.update('forward', current_time)
    #---------------
    flagXbackward = backward_counter.apply()
    if(flagXbackward):
        logger.update('backward', current_time)
    #---------------
    flagXmachine = machine_counter.apply()
    if(flagXmachine):
        logger.update('machine', current_time)
    #print(logger.stats)
if(logger.stats['total']['events'][-1]['event'][0] != 't'):
    logger.update('transition', current_time)
fvs.stop()
cv2.destroyAllWindows()
logger.save_results()