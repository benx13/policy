import cv2
from utils import *
import numpy as np
vcap = cv2.VideoCapture("rtsp://benx:benxbenx13@192.168.0.102:554/videoMain")
id = 22
i = 0
name = 'save_dir/output_vid'


widthx, heightx = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'{name}_{id}.mp4', fourcc, 23, (widthx, heightx))

while(1):
    if i == 23*60*7:
        out.release()
        id+=1
        i = 0
        widthx, heightx = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{name}_{id}.mp4', fourcc, 23, (widthx, heightx))


    ret, frame = vcap.read()
    out.write(frame)
    #cv2.imshow('VIDEO', frame)
    i+=1
    #if cv2.waitKey(1) == ord('q'):
    #    break


out.release()
cv2.destroyAllWindows()
