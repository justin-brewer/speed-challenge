import numpy as np

prec= 8
#37
labels = np.float32(np.load('labels.npy'))
labels = prec*labels

l_max = np.max(labels)
l_max = int(round(l_max)+1)
l_n = labels.shape[0]

l_ones = np.uint8(np.zeros([l_n, l_max]))
i = 0

for l in labels:
	li = int(round(l))
	l_ones[i, li] = 1
	i += 1
#37
lab_name = ('train_labels_%d_one_hot_%dxprec.npy' % (labels.shape[0], prec))
np.save(lab_name, l_ones)
