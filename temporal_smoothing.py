import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

#origin video
inputPath = Path(sys.argv[1])
cap = cv2.VideoCapture(sys.argv[1])
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame_cnt: "+str(frame_cnt))
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: "+str(fps))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("width: "+str(width)+", height: ", str(height))

#new video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
new_writer = cv2.VideoWriter('./output/TS_{:s}.mp4'.format(inputPath.stem), fourcc, fps, (width, height))

if frame_cnt==0:
    print("no frame")
    exit(0)

frames = list()

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cnt = len(frames)
kernel = [-2,-1,0,1,2]
filter_rgb = list()
for i in range(cnt):
    new_f = np.zeros((480,640,3))
    for idx,val in enumerate(kernel):
        if i+val < 0:
            new_f = new_f+frames[i-val]
        elif i+val >= cnt:
            new_f = new_f+frames[i-val]
        else:
            new_f = new_f+frames[i+val]
    new_f = np.uint8(new_f/5.0)
    new_f = cv2.cvtColor(new_f,cv2.COLOR_BGR2HSV)
    filter_rgb.append(new_f)
    #cv2.imwrite('newframe.jpg',new_f)
    #new_writer.write(new_f)

kernel_hsv = [-3,-2,-1,0,1,2,3]

bar = tqdm(total = cnt)

for i in range(cnt):
    new_f = np.zeros((480,640))
    for idx, val in enumerate(kernel_hsv):
        if i+val < 0:
            new_f = new_f+filter_rgb[i-val][:,:,2]
        elif i+val >= cnt:
            new_f = new_f+filter_rgb[i-val][:,:,2]
        else:
            new_f = new_f+filter_rgb[i+val][:,:,2]

    new_f = np.uint8(new_f/7.0)
    new_hsv = filter_rgb[i]
    new_hsv[:,:,2] = new_f
    new_bgr = cv2.cvtColor(new_hsv,cv2.COLOR_HSV2BGR)
    new_writer.write(new_bgr)
    bar.update(1)

        
cap.release()
new_writer.release()
