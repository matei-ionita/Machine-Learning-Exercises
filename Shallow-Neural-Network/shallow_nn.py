import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

N = 7000 # Cardinality of train/test data
r = 0.8 # Fraction of train data

def activation(z):
	'''
	Applies an activation function to the input z
	Currently the activation function used is the sigmoid
	'''
	a = 1/ (1+np.exp(-z))
	return a

def derivative_activation(z):
	'''
	Applies the derivative of the activation function to the input z
	Currently the activation function is the sigmoid
	'''
	return activation(z)*(1-activation(z))

def initialize(n_features):
	'''
	Initialzies the weights randomly from a multivariate normal, and the intercepts with zeros
	'''
	w_scale = 0.02
	np.random.seed(0) # Seed random generator for reproducibility
	weights_1 = np.random.randn(n_hidden,n_features) * w_scale
	b_1 = np.zeros((n_hidden,1))
	weights_2 = np.random.randn(n_hidden,1) * w_scale
	b_2 = 0

	return weights_1, b_1, weights_2, b_2



def propagate(weights_1, b_1, weights_2, b_2, features, labels):
	# A method which implements forward and backward propagation,
	# to obtain the gradient of the cost function with respect to the weights and intercepts

	# weights_1, grad_weights_1 have shape (n_hidden, n_features)
	# weights_2, grad_weights_2 have shape (n_hidden,1)
	# hidden_output has shape (n_hidden,n_samples)
	# output has shape (1,n_samples)

	n_samples = features.shape[0]

	# Forward propagation: compute the output and cost function
	hidden_output = activation( np.dot(weights_1,features.T) + b_1 )
	output = activation( np.dot(weights_2.T,hidden_output) + b_2)

	cost = - np.dot(np.log(output), labels) - np.dot(np.log(1-output), 1-labels) # Logistic cost function
	cost = np.squeeze(cost)/n_samples

	#Back propagation: compute the gradients
	grad_output =  output.T - labels 
	grad_weights_2 = np.dot(hidden_output, grad_output) / n_samples
	grad_b2 = np.squeeze( np.sum(grad_output) / n_samples )

	grad_hidden_output = np.multiply(np.dot(weights_2, grad_output.T), derivative_activation(hidden_output))
	grad_weights_1 =  np.dot(grad_hidden_output, features) /n_samples
	grad_b1 = np.sum(grad_hidden_output, axis = 1, keepdims = True) /n_samples

	return grad_weights_1, grad_b1, grad_weights_2, grad_b2, cost


def gradient_descent(train_images, train_labels, weights_1, b_1, weights_2, b_2, steps, rate):
	# A method which updates the weights and intercepts of the neural network, using gradient descent
	# Each iteration calls the propagate method, which performs forward and backward propagation to obtain the gradients
	cost_history = []
	for step in range(steps):
		grad_weights_1, grad_b1, grad_weights_2, grad_b2, cost = propagate(weights_1,b_1,weights_2,b_2,train_images,train_labels)
		weights_1 -= grad_weights_1 * rate
		weights_2 -= grad_weights_2 * rate
		b_1 -= grad_b1 * rate
		b_2 -= grad_b2 * rate
		cost_history.append(cost)

	return weights_1, b_1, weights_2, b_2, cost_history


def predict(weights_1, b_1, weights_2, b_2, features):
	# A method that predicts the label of the given test data, based on the learned weights
	# Currently only works with two classes
	hidden_output = activation( np.dot(weights_1,features.T) + b_1 )
	output = activation( np.dot(weights_2.T,hidden_output) + b_2)
	output[output<=0.5] = 0
	output[output>0.5] = 1

	return output

def show_image(img,label):
	# A method that displays an image from the MNIST dataset
	# Input: img, a numpy array of shape (1,784) containing the pixels of the image
	# and label a digit representing the true label of the image
	img=img.reshape((28,28))
	plt.imshow(img,cmap='gray')	#Make a grayscale image from the pixel array
	plt.title(label)
	plt.show()



if __name__ == '__main__':
	original_data = pd.read_csv("train.csv")
	print("Done reading")
	data = original_data.loc[original_data['label'] <= 1] # To make a binary classification problem, select only the classes 0 and 1

	images = data.iloc[1:N,1:] # Split the features from the labels
	labels = data.iloc[1:N,:1]
	images = images/255 # Standardize the data, by making the pixels take values between 0 and 1
	train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=r, random_state=0) # Split the data intro train and test samples

	n_hidden = 80 # Number of nodes in hidden layer
	steps = 400 # Number of gradient descent steps
	rate = 0.05 # Learning rate for gradient descent

	for n_hidden in range(80,90,10):
		weights_1, b_1, weights_2, b_2 = initialize(train_images.shape[1]) # Initialize the weights (randomly) and intercepts (with zeros)
		weights_1,b_1,weights_2,b_2, cost_history = gradient_descent(train_images, train_labels, weights_1,b_1,weights_2,b_2,steps,rate)
		results = predict(weights_1,b_1,weights_2,b_2,test_images)

		# Plot the evolution of cost, versus gradient descent iteration
		plt.plot(cost_history)
		plt.ylabel("Cost")
		plt.show()


		# Compare the predictions with the true labels of the test data, and count the mismatches
		errors = int(np.sum( np.abs(results.T - test_labels) ))
		print("With " + str(N) + " train/test data and " + str(n_hidden) + " hidden nodes, accuracy is " + str(1-errors/test_labels.shape[0]))



	# View misclassified images and their labels
	# Clumsily implemented, figure out how to vectorize
	lis = []
	for i in range(test_labels.shape[0]):
		if np.abs(results.T[i] - test_labels.iloc[i,0]) > 0:
			lis.append(i)
	error_images = test_labels.iloc[lis]

	for i in range(errors):
		img = test_images.loc[ error_images.index[i] ]
		show_image( img.values ,error_images.iloc[i]['label'])


	# Output results into csv file, using pandas
	'''
	df = pd.DataFrame(results)
	df.index.name='ImageId'
	df.index+=1
	df.columns=['Label']
	df.to_csv('results.csv', header=True)
	'''