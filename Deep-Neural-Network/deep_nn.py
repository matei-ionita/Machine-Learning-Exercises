import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

N = 2000 # Cardinality of train/test data
r = 0.8 # Fraction of train data
hidden_activation = "relu"
lambd = 210 # Regularization parameter
steps = 200 # For gradient descent
learning_rate = 0.33 # For gradient descent
K = 10 # number of classes
hidden_layer_dims = [93,101,73] # number of nodes on each hidden layer (not counting input and output layers)

def activate(z,activation):
	'''
	Applies an activation function to the input z
	'''
	if activation == "sigmoid":
		a = 1/ (1+np.exp(-z))
	if activation == "relu":
		a = np.maximum(0,z)
	if activation == "softmax":
		a = np.exp(z)
		s = np.sum(a,axis = 0, keepdims = True )
		a = a / s

	return a

def derivative_activate(z,activation):
	'''
	Applies the derivative of the activation function to the input z
	'''
	if activation == "sigmoid":
		a = activate(z,"sigmoid")
		der = np.multiply(a,1-a)
	if activation == "relu":
		der = np.heaviside(z,0)
	if activation == "softmax":
		n_classes = z.shape[0] # Number of classes
		n_samples = z.shape[1]
		der = np.zeros((n_classes,n_classes,n_samples))
		for i in range(n_samples):
			a = activate(z[:,i],"softmax").reshape(n_classes,1)
			der[:,:,i] = np.diagflat(a) - np.dot(a, a.T)
	return der

def initialize_params(layer_dims):
	# Initializes the weights W and intercepts b, and returns them in the dictionary params
	# shape of Wl is (layer_dims[l],layer_dims[l-1])
	# shape of bl is (layer_dims[l],1)

	np.random.seed(0) # seed random generator, for reproducibility
	scale = 2 / np.sqrt(layer_dims ) # to avoid exploding gradients, want variance of the initialized weights to be 2/n[l-1]
	# print(scale)
	params = {}
	L = len(layer_dims) 
	for l in range(1,L):
		W = np.random.randn(layer_dims[l],layer_dims[l-1]) * scale[l-1]
		b = np.zeros((layer_dims[l],1))
		params['W'+str(l)] = W
		params['b'+str(l)] = b
	return params

def forward_propagate(params,X):
	A = X
	caches = []
	L = len(params) // 2
	for l in range(1,L):
		W = params['W'+str(l)]
		b = params['b'+str(l)]
		Z = np.dot(W,A) + b
		cache = (W,b,Z,A)
		A = activate(Z,hidden_activation)
		caches.append(cache)

	# For the final layer, use softmax activation instead	
	W = params['W'+ str(L)]
	b = params['b'+ str(L)]
	Z = np.dot(W,A) + b
	cache = (W,b,Z,A)

	A = activate(Z,"softmax")
	caches.append(cache)

	return A, caches


def compute_cost(A,Y,params):
	assert(A.shape == Y.shape)
	n_samples = Y.shape[1]
	cost = - np.sum( np.multiply(Y[Y>0],np.log(A[Y>0])) ) # cross-entropy
	assert(cost.shape == () )

	L = len(params) // 2
	for l in range(1,L+1):
		cost += (lambd/2) * np.linalg.norm(params['W'+str(l)]) ** 2 # L2 regularization term
	cost /= n_samples

	return cost


def back_propagate(params,caches,A,Y):
	L = len(caches)
	n_samples = Y.shape[1]
	grads = {}

	# Initialize grad_A
	grad_A = np.zeros(Y.shape)
	grad_A [Y > 0] -= np.divide(Y[Y>0],A[Y>0]) 

	# Output layer, softmax activation
	cache = caches[L-1]
	W,Z,A_prev = cache[0], cache[2], cache[3]

	softmax_derivative = derivative_activate(Z,"softmax")

	grad_Z = np.zeros(A.shape)
	for i in range(n_samples):
		grad_Z[:,i] = np.dot(softmax_derivative[:,:,i], grad_A[:,i])

	grad_W = np.dot(grad_Z,A_prev.T) / n_samples
	grad_b = np.sum(grad_Z,axis=1, keepdims = True) / n_samples
	grad_A = np.dot( W.T,grad_Z)

	grads['W'+str(L)] = grad_W
	grads['b'+str(L)] = grad_b

	# Hidden layers, ReLU activation
	for l in range(L-1,0,-1):
		cache = caches[l-1]
		W,Z,A_prev = cache[0], cache[2], cache[3]

		grad_Z = np.multiply(grad_A, derivative_activate(Z,hidden_activation))
		grad_W = np.dot(grad_Z,A_prev.T) / n_samples
		grad_b = np.sum(grad_Z,axis=1, keepdims = True) / n_samples
		grad_A = np.dot( W.T,grad_Z )

		grads['W'+str(l)] = grad_W
		grads['b'+str(l)] = grad_b

	return grads


def update_params(params,grads,learning_rate,n_samples):
	L = len(params) // 2
	for l in range(1,L+1):
		params['W'+str(l)] = params['W'+str(l)] * (1- lambd * learning_rate / n_samples) - grads['W'+str(l)] * learning_rate # includes L2 regularization term
		params['b'+str(l)] = params['b'+str(l)]  - grads['b'+str(l)] * learning_rate
	
	return params	



def learn_model(X,Y,layer_dims,n_classes):
	# trains the neural network
	# returns the weights and intercepts in the dictionary params
	params = initialize_params(layer_dims)
	n_samples = X.shape[1]

	costs = []

	# Gradient descent
	for i in range(steps):
		# print("After " + str(i) + " steps:")
		A, caches = forward_propagate(params,X)
		grads = back_propagate(params,caches,A,Y)
		# if i % 50 == 1:
			# check_grads(grads,params,X,Y)
		params = update_params(params,grads,learning_rate,n_samples)
		if i % 10 == 0:
			cost = compute_cost(A,Y,params)
			costs.append(cost)
			print("After "+ str(i) + " iterations, the cost is " + str(cost))

	hidden_layers = layer_dims[1:len(layer_dims)-1]
	print("With "+ str(N) + " data samples, " + str(steps) + " steps, " + str(learning_rate) + " learning rate, " + str(lambd) + " regularization and layer sizes: " + str(hidden_layers))
	return params, costs


def cost_display(costs):
	x_data = 10 * np.array(range(len(costs)))
	plt.plot(x_data, costs)
	plt.ylabel("Cost")
	plt.xlabel("Step")
	plt.show()


def predict(params, X):
	A, caches = forward_propagate(params,X)
	results = np.argmax(A,axis = 0)
	return results


def read_data():
	data = pd.read_csv("train.csv")
	features = data.iloc[1:N,1:] # Split the features from the labels
	labels = data.iloc[1:N,:1]
	features = features/255 # Standardize the data, by making the pixels take values between 0 and 1

	challenge_data=pd.read_csv('test.csv') # For Kaggle Digit Recognition, read challenge data
	challenge_data/=255

	return features, labels, challenge_data


def process_data(features, labels, challenge_data):
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=r, random_state=0) # Split the data intro train and test samples


	pca = PCA(n_components=200, svd_solver = "full")
	train_features_reduced = pca.fit_transform(train_features) # Use PCA to select the most relevant features
	test_features_reduced = pca.transform(test_features)

	challenge_data_reduced = pca.transform(challenge_data)
	challenge_data_reduced = challenge_data_reduced.T

	train_features_reduced, test_features_reduced, train_labels, test_labels = train_features_reduced.T, test_features_reduced.T,train_labels.T, test_labels.T	

	n_samples_train = train_labels.shape[1]
	n_samples_test = test_labels.shape[1]

	Y_train = np.zeros((K,n_samples_train))
	Y_test = np.zeros((K,n_samples_test))
	for i in range(n_samples_train):
		Y_train[train_labels.iloc[0,i] , i] = 1
	for i in range(n_samples_test):	
		Y_test[test_labels.iloc[0,i] , i] = 1

	return train_features_reduced, Y_train, test_features_reduced, train_labels, test_labels, challenge_data_reduced


def results_to_file(results):
	df = pd.DataFrame(results)
	df.index.name='ImageId'
	df.index+=1
	df.columns=['Label']
	df.to_csv('results.csv', header=True)
	

if __name__ == "__main__":
	features, labels, challenge_data = read_data()
	print("Done reading!")
	train_features_reduced, Y_train, test_features_reduced, train_labels, test_labels, challenge_data_reduced = process_data(features, labels, challenge_data)
	print("Done processing!")

	layer_dims = [train_features_reduced.shape[0]] + hidden_layer_dims + [K] # dimensions for input, hidden and output layers
	params, costs = learn_model(train_features_reduced, Y_train, layer_dims, K) # training
	prediction_train = predict(params,train_features_reduced) # predictions on training data
	prediction_test = predict(params,test_features_reduced) # predictions on testing data


	train_correct = np.sum(prediction_train == train_labels.values)
	test_correct = np.sum(prediction_test == test_labels.values)
	print ("Training accuracy is " + str(train_correct / train_labels.shape[1]))
	print ("Testing accuracy is " + str(test_correct / test_labels.shape[1]))

	cost_display(costs) # plot the evolution of the cost function, to assist with hyperparameter selection, early stopping etc

	results=predict(params,challenge_data_reduced) # Make predictions on challenge data
	results_to_file(results) # output predictions of challenge data to csv file