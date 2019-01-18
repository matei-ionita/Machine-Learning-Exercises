import numpy as np 
import sys

k = 10 #Number of classes
d = 10 #Number of features
m = 50 #Number of training samples from each class

if __name__ == "__main__":
	# h = open("X_train.csv","w")
	# g = open("y_train.csv","w")

	X_train = np.array([]).reshape(0,d)
	y_train = np.array([]).reshape(0,1)

	# np.random.seed(0) #Seed the random generater for reproducibility
	for i in range(k):
		mean = np.random.random(d) #Randomly generate the mean and standard deviation for each class
		# std = np.random.randint(1,10) 
		std = 0.00005
		cov = std * np.eye(d)
		X = np.random.multivariate_normal(mean,cov,m) #Generate m datapoints for the current class
		y = np.array([i]*m).reshape(m,1)
		X_train = np.vstack([X_train,X])
		y_train = np.vstack([y_train,y])

	np.savetxt("X_train.csv", X_train, fmt="%.6f", delimiter=",") 
	np.savetxt("y_train.csv", y_train, fmt="%d", delimiter=",")