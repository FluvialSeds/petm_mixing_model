'''
Functions to perform 3 end-member mixing model for Inglis et al. Nat. Geosci.
Based on Optimal Multiparameter Analysis

Author:   Jordon D. Hemingway
Created:  14.04.2023
Modified: 07.02.2026
'''

#import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy import ma, eye
from numpy.linalg import inv, norm
from numpy.random import uniform
from scipy.optimize import nnls

#make functions

#function for non-negative least squares (NNLS) end-member mixing analysis (EMMA)
def nnls_emma(Adf, bdf, whiten = True, stu_wt = 1):
	'''
	Calculates end-member contributions when A matrix contains conservative
	tracers that do not apply to every end-member. Inputted bdf matrix must
	explicitly contain the sum-to-unity constraint (due to MC implementation).

	Parameters
	----------
	Adf : pd.DataFrame
		[nT+1 x nEM] design matrix, where nT is the number of conservative
		tracers and nEM is the number of end-members. A must contain NaNs where
		conservative tracers do not apply. Final row is the sum-to-unity 
		constraint. Row names will be carried through to final result fractional 
		contribution names.

	bdf : pd.DataFrame
		[nT+1 x nSam] data matrix, where nT is the number of conservative
		tracers and nSam is the number of samples in the dataset. Final row is 
		the sum-to-unity constraint. Row names must match those of A. Column 
		names will be carried through to final result.

	whiten : bool
		Tells the function whether or not to whiten the data when fitting, i.e.,
		to subtract the mean of A and divide by std. dev. of A in order to get
		all tracers equally weighted.

	stu_wt : int or float
		If `whiten = True', tells the function how much more strongly to weight
		the sum-to-unity constraint relative to all other tracers. This
		becomes important when nT+1 > nEm, since the sum-to-unity constraint
		is then just treated as any other tracer in the least-squares regression.
		Defaults to unity.

	Returns
	-------
	res : pd.dataframe
		[nEM x nSam] dataframe of resulting fractional contribution values.

	rmse : float
		Resulting root-mean square error for the nnls solution.
	'''

	#make copies so that nothing gets changed outside the function
	A = Adf.copy(deep = True)
	b = bdf.copy(deep = True)

	#get shapes
	_, nEm = np.shape(A)
	nTp1, nSam = np.shape(b)

	#if whiten, then do so:
	if whiten:

		#calculate mean and std. dev.
		Abar = A.mean(axis=1)
		s = A.std(axis=1)

		#subtract and divide (keeping final row untouched for STU)
		A.iloc[:-1,:] = A.iloc[:-1,:].sub(
			Abar[:-1], axis = 0).div(
			stu_wt*s[:-1], axis = 0
			)

		b.iloc[:-1,:] = b.iloc[:-1,:].sub(
			Abar[:-1], axis = 0).div(
			stu_wt*s[:-1], axis = 0
			)

	#subtract 1 from b STU constraint (since we will subtract this from A STU
	# constraint)
	b.loc['stu'] = b.loc['stu'] - 1

	#get b to [nEm*nSam x 1] vector
	bvec = np.reshape(b.T, nTp1*nSam)

	#get A to [nTp1*nSam x nEm*nSam] block diagonal array
	I = np.eye(nSam, dtype = float)
	I[I == 0] = np.nan #nan-fill so that these stay empty after subtraction
	Abd = np.kron(I, A)

	#now subtract b vec from Abd
	x = Abd - bvec[:,None]
	X = np.nan_to_num(x,0) #re-fill nans as zero

	#make [nT*0,1] array
	z = np.zeros(nTp1)
	z[-1] = 1 #keep STU constraint as unity
	y = np.tile(z, nSam)

	#solve
	res, rmse = nnls(X, y)
	resvec = res.reshape([nSam,nEm])

	r = pd.DataFrame(resvec, index = b.columns, columns = A.columns)

	return r, rmse

#function to perform Monte Carlo masked NNLS EMMA
def nnls_emma_MC(Adfmin, Adfmax, bdf, nIter = 1000, stu_err = 0.05,
frac_save = 0.1, whiten = True, stu_wt = 1):
	'''
	Calculates end-member contributions in Monte Carlo fashion when A matrix 
	contains conservative tracers that do not apply to every end-member. Here, 
	inputted A and bdf matrices do not explicitly contain the sum-to-unity
	constraint since it is implemented in each Monte Carlo iteration.

	Parameters
	----------
	Adfmin : pd.DataFrame
		[nT x nEM] design matrix containing minimum values of each tracer for 
		each end member, where nT is the number of conservative tracers and nEM 
		is the number of end-members. A must contain NaNs where conservative 
		tracers do not apply. Row names will be carried through to final result 
		fractional contribution names.

	Adfmax : pd.DataFrame
		[nT x nEM] design matrix containing maximum values of each tracer for 
		each end member, where nT is the number of conservative tracers and nEM 
		is the number of end-members. A must contain NaNs where conservative 
		tracers do not apply. Row names will be carried through to final result 
		fractional contribution names.

	bdf : pd.DataFrame
		[nT x nSam] data matrix, where nT is the number of conservative tracers
		and nSam is the number of samples in the dataset. Row names must match
		those of A. Column names will be carried through to final result.

	nIter : int
		Number of Monte Carlo iterations to perform; defaults to 1000.

	stu_err : float
		Allowable fractionation offset from the sum-to-unity constraint. For
		example, stu_err = 0.05 allows for solutions that sum to 0.95 - 1.05.
		Defaults to 0.05.

	frac_save : float
		Fraction of best-fitting Monte Carlo solutions to save for final
		statistics. Defaults to 0.1 (i.e., retain top 10 percent).

	whiten : bool
		Tells the function whether or not to whiten the data when fitting, i.e.,
		to subtract the mean of A and divide by std. dev. of A in order to get
		all tracers equally weighted.

	stu_wt : int or float
		If `whiten = True', tells the function how much more strongly to weight
		the sum-to-unity constraint relative to all other tracers. This
		becomes important when nT+1 > nEm, since the sum-to-unity constraint
		is then just treated as any other tracer in the least-squares regression.
		Defaults to unity.

	Returns
	-------
	rvg : np.ndarray
		[nSam x nEm x niter*frac_save] results matrix containing the calculated
		fractional contributions of each end member to each sample for each of
		the retained Monte Carlo iterations.

	rmseg : np.ndarray
		Length niter*frac_save array of resulting root-mean square error for
		each fo the retained Monte Carlo iterations.

	Avg : np.ndarray
		[nTr x nEm x niter*frac_save] design matrix containing the calculated
		end-member compositions for each of the retained Monte Carlo iterations.
	'''

	#make everything copies to avoid changing outside of function
	Amin = Adfmin.copy(deep = True)
	Amax = Adfmax.copy(deep = True)
	# bi = bdf.copy(deep = True)

	#get shapes
	nT, nEm = np.shape(Amin)
	_, nSam = np.shape(bdf)

	#make arrays for storing
	resvec = np.zeros([nSam, nEm, nIter])
	Avec = np.zeros([nT, nEm, nIter])
	rmsevec = np.zeros(nIter)

	#loop through each iteration, making a new A and b matrices and solving
	for i in range(nIter):

		#make A between Amin and Amax and STU between 1-stu_err and 1+stu_err
		A = Adfmin + (Amax - Amin)*uniform(size = np.shape(Amin))
		stu = np.ones(nEm) + uniform(low = -stu_err, high = stu_err, size = nEm)

		#add STU constraint to matrices
		A.loc['stu'] = stu
		
		b = bdf.copy(deep = True)
		b.loc['stu'] = np.ones(nSam)

		#run masked solver
		res, rmse = nnls_emma(A, b, whiten = whiten, stu_wt = stu_wt)

		#store in array
		resvec[:,:,i] = res.values
		rmsevec[i] = rmse
		Avec[:,:,i] = A.values[:-1,:]

	#for each iteration, only keep solutions with the lowest frac_save percentile
	# of rmse

	#cutoff value
	co = np.quantile(rmsevec, frac_save)
	coind = np.where(rmsevec <= co)[0]

	#save only good fitting solutions
	rvg = resvec[:,:,coind]
	rmseg = rmsevec[coind]
	Avg = Avec[:,:,coind]

	return rvg, rmseg, Avg

if __name__ == '__main__':

	#------------------------------------------#
	# MAKE ARTIFICAL DATASET AND TEST UNMIXING #
	#------------------------------------------#

	#set path
	path = '../02 input data/'
	fig_path = '../04 output figures/'

	#input scalars
	nEm = 3 #number of end-members
	nTr = 3 #number of tracers measured
	nSam = 250 #number of samples

	#import A matrix
	Adf = pd.read_csv(path+'A_test.csv',
		index_col = 0,
		)

	#ensure floats
	Adf = Adf.astype(float)

	#fractional contributions matrix
	f = np.random.uniform(
		low = 0, 
		high = 1,
		size = [nEm,nSam]
		)

	fs = f / f.sum(axis=0)[None,:]
	fdf = pd.DataFrame(fs, index = Adf.columns)

	#manually go through and calculate "measured" data. Can't do this by normal
	# matrix multiplication since some entries in A are np.nan and because
	# a masked array inherently converts these to zero
	# (this is admittedly not clever but I can't think of a better way...)

	#make empty dataframe to store results
	bdf = pd.DataFrame(np.zeros([nTr+1, nSam]), index = Adf.index)

	if 'TAR' in bdf.index:
		bdf.loc['TAR',:] = Adf.loc['TAR','algal'] * \
			fdf.loc['algal',:]/(fdf.loc['algal',:] + fdf.loc['plant',:]) + \
			Adf.loc['TAR','plant'] * fdf.loc['plant',:]/(fdf.loc['algal',:] + \
			fdf.loc['plant',:])

	if 'BIT' in bdf.index:
		bdf.loc['BIT',:] = Adf.loc['BIT','algal'] * \
			fdf.loc['algal',:]/(fdf.loc['algal',:] + fdf.loc['soil',:]) + \
			Adf.loc['BIT','soil'] * fdf.loc['soil',:]/(fdf.loc['algal',:] + \
			fdf.loc['soil',:])

	if 'CN' in bdf.index:
		#project onto nEm x nSam matrix
		cn = Adf.loc['CN'].values
		bdf.loc['CN',:] = np.inner(cn[None,:],fs.T)

	#finally add STU
	bdf.loc['stu',:] = np.ones(nSam)

	#add noise to calculated data
	#assumed noise for each end member (TAR, BIT, C/N, stu)
	sigma = np.array([0.05, 0.05, 1, 0.01])
	noise = np.random.randn(nTr+1,nSam)*np.outer(sigma,np.ones(nSam))

	#make noisy b dataframe
	nbdf = bdf + noise

	#now predict and compare
	res, rmse = nnls_emma(
		Adf, 
		nbdf, 
		whiten = True, 
		stu_wt = 10
		)

	#make plot
	fig, ax = plt.subplots(1,1)

	for i,em in enumerate(res):

		plt.scatter(fs[i,:], res[em], label = em)

	ax.plot([0,1],[0,1], c = 'k', linewidth = 2, label = '1:1 line')

	ax.legend(loc = 'best')

	ax.set_xlabel('true fractional contributions')
	ax.set_ylabel('calculated fractional contributions')

	ax.set_xlim([-0.05,1.05])
	ax.set_ylim([-0.05,1.05])

	fig.savefig(fig_path + 'training_set_result.pdf')
