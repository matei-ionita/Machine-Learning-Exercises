import numpy as np
import sys

#Run as: python clustering.py wine.csv

#Set the meta-parameters: K is the number of clusters, T the number of iterations
K, T = 3, 10


#A method that evaluates a Gaussian function, with given mean and covariance, at the argument x
def gaussian(x, mean, cov):
	d = len(x)
	v = x - mean
	v = v.reshape(d,1)
	var = np.linalg.inv(cov)
	exponent = np.dot(np.dot(np.transpose(v),var),v)
	prefactor = 2* np.pi * np.linalg.det(cov)
	return prefactor**(-1/2) * np.exp(-1/2 * exponent)


#The K-means algorithm for clustering
def KMeans(data,n):
	np.random.seed(0) #For reproducible results, seed the random generator
	centroids = data[np.random.randint(n, size = K)]	#To initialize centroids, pick K datapoints uniformly at random

	c = np.zeros(n)	#Will hold cluster assignments for all datapoints

	'''
	We look for a local minimum of the loss function using coordinate descent. For each iteration t,
	first update cluster assignments, keeping the centroids fixed,
	then update centroids, keeping the cluster assignments fixed.
	'''
	for t in range(T):
		#Update cluster assignments
		for i in range(n):
			minimum = float('inf')
			for k in range(K):
				distance = np.linalg.norm(data[i]-centroids[k])
				if  distance < minimum:
					minimum = distance
					c[i] = k

		#Update centroids
		for k in range(K):
			cluster_k = np.nonzero(c == k)[0]

			nk = len(cluster_k)
			centroids[k] = np.sum(data[cluster_k][:],axis=0)
			centroids[k]/= nk

	#Output the final cluster assignments
	np.savetxt("kmeans_assignments.csv", c, delimiter="\n")


#Gaussian Mixture Model (GMM) clustering, using the Expectation-Maximization algorithm (EM)
def EMGMM(data,n):
	d = len(data[0])

	#Initialize cluster priors and cluster likelihoods
	pi = 1/K * np.ones(K)	#Initial cluster prior is uniform
	np.random.seed(0)	#For reproducible results, seed the random generator
	mu = data[np.random.randint(n, size = K)]	#Initial cluster likelihood is Gaussian, centered on a datapoint sampled uniformly at random
	sigma = [np.eye(d)]*K 						#Covariance is identity matrix

	phi = np.zeros( (n,K) )	#phi[i,k] will hold the probability that datapoint i belongs to cluster k

	for t in range(T):
		'''
		Expectation step: the probability to assign datapoint i to cluster k is proportional to the cluster prior of k,
		and to the cluster likelihood
		'''
		for i in range(n):
			for k in range(K): 
				phi[i,k] = pi[k] * gaussian(data[i],mu[k],sigma[k])	#Outsource Gaussian computation to a separate method
			s = np.sum(phi[i,:])
			phi[i,:] /= s 	#Normalize the probability distribution

		#Maximization step
		for k in range(K):
			nk = np.sum(phi[:,k])
			pi[k] = nk/n #Update cluster prior with the fraction of datapoints currently assigned to it

			partial_mu = np.zeros(d)
			for i in range(n): 
				partial_mu += phi[i,k] * data[i]
			mu[k,:] = partial_mu/nk #Update mean for current cluster with the weighted average of datapoints currently assigned to it

			partial_sigma = np.zeros((d,d))
			for i in range(n): 
				v = (data[i] - mu[k]).reshape(d,1)
				partial_sigma += phi[i,k] * np.dot(v,np.transpose(v))
			sigma[k] = partial_sigma / nk #Update covariance for current cluster

	c = np.argmax( phi,axis=1 ) #For each datapoint, find the cluster it most likely belongs to, according to the distribution phi
	np.savetxt("gmm_assignments.csv", c, delimiter="\n")

		

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Please run as: python "+ sys.argv[0] + " dataset_filename")
	else:
		X = np.genfromtxt(sys.argv[1], delimiter = ",") #Read data from file
		n = len(X)
		features = [1,2,3,4,6,7,8,9,10,11,12]	#Select features (columns) to do training on
		X = X[:,features]

		KMeans(X,n)
		EMGMM(X,n)