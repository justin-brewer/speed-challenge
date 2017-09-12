import numpy as np

#37
data = np.load('test_10798x2400_flow_zoom.npy')
n = data.shape[0]

new_dat = np.empty_like(data)

# window size to average optic flow data
avg_win = 5

new_dat[0:avg_win] = np.mean(data[0:avg_win], axis=0)
new_dat[-avg_win:] = np.mean(data[-avg_win:], axis=0)

for i in range(avg_win, n-avg_win):
	new_dat[i,...] = np.mean(data[i-avg_win:i+1], axis=0)

#37
fn = ('test_%dx2400_fzm_%davg.npy' % (n, avg_win))
np.save(fn, new_dat)
