import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import sys

k = 10 #Number of classes
M = 5000 #Number of samples to train on


def naiveBayes(X_train, y_train, X_test):
	#Naive Bayes fits the data on k Gaussian clusters, using Bayesian inference.

	n = len(y_train)
	m,d = np.shape(X_test)

	# The class prior is pi, a discrete distribution
	# We model the class likelihood as a normal distribution, with mean and cov
	pi = np.zeros(k)
	mean = np.zeros((k,d))
	cov = np.zeros((k,d,d))
	var = np.zeros((k,d,d))
	prefactor = np.zeros(k)
	for current_class in range(k):
		points = X_train[ [i for i in range(n) if y_train[i]==current_class] ] #Select points labeled with current class
		pi[current_class] = points.shape[0] # Class prior is number of data points in current class; will normalize later

		mean[current_class] = np.sum(points,axis=0,keepdims=True) # The mean for class likelihood is the average over samples in the class
		mean[current_class] /= pi[current_class]

		v = points - mean[current_class] # Center data around mean

		for row in v:
			cov[current_class] += np.outer(row,row) # Compute the covariance - figure out how to vectorize this
		cov[current_class] /= pi[current_class]

		var[current_class] = np.linalg.inv(cov[current_class]) # Invert covariance to obtain variance
		prefactor[current_class] = (2 * np.pi * np.linalg.det(cov[current_class]) )**(-1/2) # Compute the prefactor of the Gaussian function

	pi/=n # Normalization of class priors

	# print(pi)
	# print(prefactor)

	
	# Compute probabilities of labels for test data
	prob = np.zeros((k,m)) # Will hold the probability that test point i belongs to class j
	exponent = np.zeros(m)
	for current_class in range(k):
		v = X_test - mean[current_class] # Center data around mean
		for i in range(m):
			exponent[i] =  - np.dot(v[i],v[i]) # Figure out how to vectorize this
		prob[current_class] = pi[current_class] * prefactor[current_class] * np.exp( exponent ) # Using Bayes' rule: posterior is proportional to prior * likelihood


	row_sums = np.sum(prob,axis=0)
	prob /= row_sums[ np.newaxis,:] # Normalize the posterior

	return prob


if __name__ == '__main__':
	from_file = np.genfromtxt(sys.argv[1], delimiter=",") # numpy's genfromtxt is slower at parsing a csv file, compared to pandas read_csv
	data = from_file[1:M:,1:]
	labels = from_file[1:M:,0]

	train_data, test_data, train_labels, test_labels = train_test_split( data, labels, test_size=0.25, random_state=42)

	sc = StandardScaler()  
	train_data = sc.fit_transform(train_data) # standardize data: important before doing PCA or LDA
	test_data = sc.transform(test_data) # transform test data using the fit parameters of the training data

	for N in range(15,16): # number of principal components

		preprocessing = "QDA" # Choose PCA, LDA or QDA

		if preprocessing == "PCA":
			pca = PCA(n_components=N, svd_solver = "full")
			train_data_reduced = pca.fit_transform(train_data) # Use PCA to select the most relevant features, and avoid singular covariance matrices
			test_data_reduced = pca.transform(test_data)

			prob = naiveBayes(train_data_reduced, train_labels, test_data_reduced) # Naive Bayes returns probabilities for each pair (test sample,class)
			prediction = np.argmax(prob, axis=0) # For now we just pick the most probable class and discard the other information
		else preprocessing == "LDA":
			lda = LDA(n_components=N)  
			train_data_reduced = lda.fit_transform(train_data, train_labels) # LDA selects the features taking labels into account
			test_data_reduced = lda.transform(test_data) 
			prediction = lda.predict(test_data) # Naive Bayes is already implemented in sklearn's LDA

		#Evaluation
		correct = np.count_nonzero( prediction == test_labels.astype(int) )
		print("Using " + str(M) + " training samples and " + preprocessing + " with " + str(N) + " components, accuracy of Naive Bayes is " + str(correct/test_labels.shape[0]) )