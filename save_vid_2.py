
import cv2
from utils import *
import numpy as np
#vcap = cv2.VideoCapture()
vcap = cv2.VideoCapture("rtsp://192.168.0.100:5543/live/channel0")
#vcap = cv2.VideoCapture(0)
id = 28
i = 0
name = 'save_dir/output_vid'


widthx, heightx = 2304, 1296
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'{name}_{id}.mp4', fourcc, 23, (widthx, heightx))

import threading
from threading import Lock
import cv2



while(1):
    ret, frame = vcap.read()
    if ret:
        out.write(frame)
        cv2.imshow('VIDEO', frame)
        i+=1

        if i == 23*60*7:
            out.release()
            id+=1
            i = 0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'{name}_{id}.mp4', fourcc, 25, (widthx, heightx))

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        print(ret)

out.release()
cv2.destroyAllWindows()
