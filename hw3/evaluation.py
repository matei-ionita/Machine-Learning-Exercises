import numpy as np
import sys

#Set the meta-variables: K is the number of clusters
K = 3

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Please run as: python " + sys.argv[0] + "dataset_filename training_result_filename")
	else:
		X = np.genfromtxt(sys.argv[1],delimiter = ",") #Read data from file
		label = X[:,0] #Assuming that first column is the label
		n = len(label)
		result = np.genfromtxt(sys.argv[2],delimiter = ",") #Read training results from file


		penalty = 0
		for k in range(1,K+1):
			l = [i for i in range(n) if label[i]==k]	#List datapoints with label k
			cluster_label = max( set(result[l]), key=list(result[l]).count ) #Find 
			for point in l:
				if result[point] != cluster_label: penalty +=1

		print(penalty)