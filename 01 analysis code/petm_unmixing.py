'''
Scripts to execute 3 end-member mixing model for Inglis et al. Nat. Geosci.
Based on Optimal Multiparameter Analysis

Author:   Jordon D. Hemingway
Created:  20.06.2023
Modified: 17.02.2026
'''

#import packages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from mixfuncs import nnls_emma
from mixfuncs import nnls_emma_MC

#set data paths
Apth = '../02 input data/'
bpth = Apth + 'bvecs/'
respth = '../03 output data/'
figpth = '../04 output figures/'

#import min and max A matrices
Atot_min = pd.read_csv(Apth+'A_tot_min.csv', index_col = 0)
Atot_max = pd.read_csv(Apth+'A_tot_max.csv', index_col = 0)

#loop through all b vectors and solve
for bfile in os.listdir(bpth):

	#catch any hidden other files (e.g., .DS_Store)
	if '.csv' not in bfile:
		continue

	#skip file with d13C values since this is not diagnostic and is not included
	# in any of the PETM sections
	elif bfile == 'Modern_GoM_all_data.csv':
		continue

	print(f'Working on file: {bfile}')

	#import b and drop rows with missing values
	bdf = pd.read_csv(bpth+bfile, index_col = 0)
	bdf = bdf.dropna().T

	#make A matrix, keeping only conservative tracers present in vector b
	Adfmin = Atot_min.T[bdf.T.columns].T.iloc[:,:-1]
	Adfmax = Atot_max.T[bdf.T.columns].T.iloc[:,:-1]

	#perform Monte Carlo unmixing
	rvg, rmseg, Avg = nnls_emma_MC(
		Adfmin, 
		Adfmax, 
		bdf, 
		nIter = 100000, 
		# nIter = 1000, 
		stu_err = 0.05,
		frac_save = 0.01,
		whiten = True,
		stu_wt = 10)

	#get number of successful iterations
	_, _, nIter = np.shape(rvg)

	#add error catching for sites with values outside possible range
	try:
		#store best-fit fractional contribution results
		ems = list(Adfmin.columns.values)
		stats = ['_median', '_25pctile', '_75pctile', '_mean', '_std']
		cols = [e+s for e in ems for s in stats] + ['nIter','rmse_mean']

		#make empty data frame
		resdf = pd.DataFrame(columns = cols, index = bdf.columns, dtype = float)

		#store med, mean, std. dev., etc.
		resdf.loc[:,[e+stats[0] for e in ems]] = np.median(rvg, axis = 2)
		resdf.loc[:,[e+stats[1] for e in ems]] = np.quantile(rvg, 0.25, axis = 2)
		resdf.loc[:,[e+stats[2] for e in ems]] = np.quantile(rvg, 0.75, axis = 2)
		resdf.loc[:,[e+stats[3] for e in ems]] = np.mean(rvg, axis = 2)
		resdf.loc[:,[e+stats[4] for e in ems]] = np.std(rvg, axis = 2)
		resdf.loc[:,'nIter'] = nIter
		resdf.loc[:,'rmse_mean'] = rmseg.mean()

		#save
		resdf.to_csv(respth+bfile[:-4]+'_frac_abund_res.csv')

		#store best-fit design matrix results
		cts = list(Adfmin.index.values)
		stats = ['_median', '_25pctile', '_75pctile', '_mean', '_std']
		cols = [c+s for c in cts for s in stats] + ['nIter']

		#make empty data frame
		Adf = pd.DataFrame(index = cols, columns = Adfmin.columns.values, dtype = float)

		#store med, mean, std. dev., etc.
		Adf.loc[[c+stats[0] for c in cts],:] = np.median(Avg, axis = 2)
		Adf.loc[[c+stats[1] for c in cts],:] = np.quantile(Avg, 0.25, axis = 2)
		Adf.loc[[c+stats[2] for c in cts],:] = np.quantile(Avg, 0.75, axis = 2)
		Adf.loc[[c+stats[3] for c in cts],:] = np.mean(Avg, axis = 2)
		Adf.loc[[c+stats[4] for c in cts],:] = np.std(Avg, axis = 2)
		Adf.loc['nIter',:] = nIter

		#save
		Adf.to_csv(respth+bfile[:-4]+'_A_matrix_res.csv')

	#catch error and continue
	except:
		continue

	#plot data
	fig, ax = plt.subplots(1,1, figsize = (3,8))

	#loop through end members and plot
	for e in ems:

		#median
		ax.plot(resdf.loc[:,e+'_median'], bdf.columns[::-1], label = e+'_median')

		#IQR
		ax.fill_betweenx(
			bdf.columns[::-1], 
			resdf.loc[:,e+'_25pctile'], 
			resdf.loc[:,e+'_75pctile'],
			alpha = 0.3)

	#add legend and label axes
	ax.legend(loc = 'best')
	ax.set_ylabel('depth')
	ax.set_xlabel('fractional abundance')
	ax.set_xlim([-0.05,1.05])
	ax.invert_yaxis()

	plt.tight_layout()

	#save results
	fig.savefig(figpth+bfile[:-4]+'_mixing_plot.pdf',
		transparent = True,
		bbox_inches = 0
		)

	plt.close('all')
