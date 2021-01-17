import cv2
import sys
from pathlib import Path
import numpy as np
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
print("width: "+str(width)+" , height: "+str(height))

#new video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
new_writer = cv2.VideoWriter('./output/LA_{:s}.mp4'.format(inputPath.stem), fourcc, fps, (width,height))

if frame_cnt==0:
    print("no frame")
    exit(0)

frames_yuv = list()

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    frame_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    frames_yuv.append(frame_yuv)

cnt = len(frames_yuv)

#y = [0,255] luminance
#u = [0,255]
#v = [0,255]

bar = tqdm(total = cnt)
new_frame = cv2.cvtColor(frames_yuv[0], cv2.COLOR_YUV2BGR)
new_writer.write(new_frame)
bar.update(1)

frame_pre = frames_yuv[0]

for i in range(1,cnt):
    new_frame = frames_yuv[i]
    for j in range(height):
        for k in range(width):
            x2 = np.square(int(frame_pre[j][k][1])-int(frames_yuv[i][j][k][1]))
            y2 = np.square(int(frame_pre[j][k][2])-int(frames_yuv[i][j][j][2]))
            xy_sqrt = np.sqrt(x2+y2)

            if xy_sqrt < 25: #same color, correct the brightness
                if np.abs(int(frames_yuv[i][j][k][0])-int(frame_pre[j][k][0])) < 25:
                    if frame_pre[j][k][0] < frames_yuv[i][j][k][0]:
                        new_frame[j][k][0] = frame_pre[j][k][0] +1
                    elif frame_pre[j][k][0] > frames_yuv[i][j][k][0]:
                        new_frame[j][k][0] = frame_pre[j][k][0] -1
    frame_pre = new_frame
    new_bgr = cv2.cvtColor(new_frame,cv2.COLOR_YUV2BGR)
    new_writer.write(new_bgr)
    bar.update(1)

                
cap.release()
new_writer.release()
