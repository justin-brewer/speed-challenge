import numpy as np
import cv2
from time import sleep
from scipy.ndimage.filters import gaussian_filter1d as gfilt


cap = cv2.VideoCapture('test.mp4')
ret, frame1 = cap.read()

speeds = np.load('predictions/test_predictions_avg123.npy')
# speeds = gfilt(speeds, 10)


i = 0
n = 10798

save_vid = False

if save_vid:
	filename = ('train_speed_%d.avi' % (n))
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(filename, fourcc, 20, (640,480), isColor=False)

while(ret and i < n):
	ret, frame2 = cap.read()
	frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

	if i > 15:
		speed = np.mean(speeds[i-3:i+1])
	else:
		speed = speeds[i]

	f_name = ('(%4.1f m/s) (%4.1f mph)' % (speed, speed*2.24))

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame2,f_name,(10,475), font, 1,(255,255,255),2)

	if save_vid:
		out.write(frame2)
	# lab_vid[i,...] = frame2

	cv2.imshow('frame', frame2)
	k = cv2.waitKey(35) & 0xff
	# sleep(0.01)
	if k == 27:
	    break
	i += 1
	cv2.destroyAllWindows()
#######################################################################
if save_vid:
	out.release()

cap.release()
