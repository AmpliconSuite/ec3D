import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
	objective_values = []
	log_fn = sys.argv[1]
	output_fn = sys.argv[2]
	fp = open(log_fn, 'r')
	for line in fp:
		s = line.strip().split()
		if "Estimated alpha" in line:
			objective_values.append([float(s[6][:-1]), float(s[9][:-1])])
		if "Log likelihood at iteration" in line:
			objective_values[-1].append(float(s[-1][:-1]))
		if "RMSD" in line:
			objective_values[-1].append(float(s[5]))
			objective_values[-1].append(float(s[8]))
	fp.close()
	value = {'round':[i+1 for i in range(len(objective_values))], \
			'obj_value':[objective_values[i][2] for i in range(len(objective_values))], \
			'alpha':[objective_values[i][0] for i in range(len(objective_values))], \
			'beta':[objective_values[i][1] for i in range(len(objective_values))], \
			'RMSD':[objective_values[i][3] for i in range(len(objective_values))], \
			'PCC':[objective_values[i][4] for i in range(len(objective_values))]}
	
	# ref: https://github.com/BindiChen/machine-learning.git
	# Create figure and axis #1
	fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
	fig.subplots_adjust(right=0.8)
	# plot line chart on axis #1
	p1, = ax1.plot(value['round'], value['obj_value']) 
	ax1.set_xscale('log')
	ax1.set_xlabel('log(round)')
	ax1.xaxis.label.set_fontsize(14)
	ax1.set_ylabel('objective value')
	ax1.legend(['objective value'], loc="upper left")
	ax1.yaxis.label.set_color(p1.get_color())
	ax1.yaxis.label.set_fontsize(14)
	ax1.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

	# set up the 2nd axis
	ax2 = ax1.twinx() 
	# plot bar chart on axis #2
	p2, = ax2.plot(value['round'], value['RMSD'], color='orange')
	ax2.set_xscale('log')
	ax2.grid(False) # turn off grid #2
	ax2.set_ylabel('RMSD')
	ax2.legend(['RMSD'], loc="upper center")
	ax2.yaxis.label.set_color(p2.get_color())
	ax2.yaxis.label.set_fontsize(14)
	ax2.tick_params(axis='y', colors=p2.get_color(), labelsize=14)

	# # set up the 3rd axis
	ax3 = ax1.twinx()
	# Offset the right spine of ax3.  The ticks and label have already been
	# placed on the right by twinx above.
	ax3.spines.right.set_position(("axes", 1.15))
	# Plot line chart on axis #3
	p3, = ax3.plot(value['round'], value['PCC'], color='red')
	ax3.grid(False) # turn off grid #3
	ax3.set_ylabel('PCC')
	ax3.legend(['PCC'], loc="upper right")
	ax3.yaxis.label.set_color(p3.get_color())
	ax3.yaxis.label.set_fontsize(14)
	ax3.spines['right'].set_visible(False)
	ax3.tick_params(axis='y', colors=p3.get_color(), labelsize=14)
	
	plt.savefig(output_fn, bbox_inches='tight')
