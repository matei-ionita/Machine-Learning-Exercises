import numpy as np
import sys

#Run as: python pmf.py ratings.csv



#Set meta-parameters
lam = 2	#Weight of the ridge terms in the loss function
sigma2 = 0.1	#Weight of the factorization error in the loss function
d = 5	#Rank of the factorization
T = 51	#Number of iterations


def PMF(train_data):
	nu = int(max(train_data[:,0]))	#Number of users
	nv = int(max(train_data[:,1]))	#Number of locations

	#Construct the rating matrix from the given ratings
	M = np.zeros((nu,nv))
	for row in train_data:
		M[ int(row[0])-1,int(row[1])-1 ] = row[2]
	np.savetxt("ratings_given.csv", M, fmt='%i', delimiter=",")


	#Initialize the user profiles U and the location profiles V
	U = np.zeros((nu,d))
	V = np.zeros((d,nv))

	np.random.seed(0)	#Seed the random generator for reproducibility of results
	V = np.random.multivariate_normal(np.zeros(d),1/lam * np.eye(d),nv).transpose()	#Initialize the location profiles with random vectors from a normal distribution

	#Initialize the loss function
	L = np.zeros
	L = - 1/2 * 1/sigma2 * np.sum( np.power(M, 2) ) - lam/2 * np.sum([np.dot(V[:,j], V[:,j]) for j in range(nv)])

	#Find a local critical point of the loss function, using coordinate descent
	for t in range(1,T):
		#Update U to extremize L, keeping V fixed
		for i in range(nu):
			U[i] = np.dot(V,M[i])

			#Compute by hand the matrix to be inverted
			s = np.zeros((d,d))
			for j in range(nv):
				if M[i,j] != 0:
					s += np.dot(V[:,j].reshape(d,1), V[:,j].reshape(1,d))
			mat = lam * sigma2 * np.eye(d) + s

			U[i] = np.dot(np.linalg.inv(mat), U[i])

		#Update V to extremize L, keeping U fixed
		for j in range(nv):
			V[:,j] = np.dot(M[:,j],U)

			#Compute by hand the matrix to be inverted
			s = np.zeros((d,d))
			for i in range(nu):
				if M[i,j] != 0:
					s += np.dot(U[i].reshape(d,1), U[i].reshape(1,d))
			mat = lam * sigma2 * np.eye(d) + s

			V[:,j] = np.dot(np.linalg.inv(mat), V[:,j])

		#Compute the loss function
		er = M - np.dot(U,V)  #The difference M - UV is the error
		for i in range(nu):
			for j in range(nv):
				if M[i,j] == 0:  #Brute-force masking of the user-location pairs for which the rating is unknown
					er[i,j] = 0
				else:
					er[i,j] = er[i,j]**2

		L = - 1/2 * 1/sigma2 * np.sum( er ) - lam/2 * np.sum([np.dot(V[:,j], V[:,j]) for j in range(nv)]) - lam/2 * np.sum( [np.dot(U[i], U[i]) for i in range(nu)] )

	return L,U,V


if __name__ == '__main__':
	train_data = np.genfromtxt(sys.argv[1], delimiter = ",")	#Read data from file
	L, U, V = PMF(train_data)	#PMF outputs the factorization UV and the loss function L

	prediction = np.round(np.dot( U,V ))
	prediction[prediction<0] = 0	#Set negative predictions to zero

	np.savetxt("ratings_predicted.csv", prediction, fmt='%i', delimiter=",")