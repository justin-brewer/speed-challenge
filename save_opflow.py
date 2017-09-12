import cv2
import numpy as np
from scipy.ndimage.interpolation import zoom
from time import sleep
import math
#37
cap = cv2.VideoCapture("test.mp4")
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

n = 20399
zf = 1/16
f_xy = np.empty([n+1, 1200, 2])

i = 1
while(i < n):
	ret, frame2 = cap.read()
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 7, 1.5, 0)

	fy = zoom(flow[...,0], [zf, zf]).reshape(1200)
	fx = zoom(flow[...,1], [zf, zf]).reshape(1200)
	f_xy[i+1, :, 0] = fy
	f_xy[i+1, :, 1] = fx
	i += 1
	if i % 100 == 0:
		print('finished: ', i)
	prvs = next

f_xy[0, :, 0] = f_xy[1, :, 0]
f_xy[0, :, 1] = f_xy[1, :, 1]

f_xy = f_xy.reshape(n+1, 2400)
#37
file_name = (('data_%dx2400_flow_zoom.npy') % (i+1))
np.save(file_name, f_xy)

cap.release()
