# Machine-Learning-Exercises

A collection of ML algorithms that I implemented from scratch in Python. 
The goal is to familiarize myself with the algorithms, rather than to implement them as efficiently or elegantly as possible.

The Naive Bayes, Clustering and Probabilistic Matrix Factorization implementations are inspired from the programming assignments
of the ColumbiaX course "Machine Learning". I added extra preprocessing steps, other algorithms for comparison,
and experimented with them on real-world data sets.

The outline of the Neural Network implementations is inspired from the deeplearning.ai course "Neural Networks and Deep Learning".
I changed various implementation details, extended them from binary classifiers to multiclass classifiers, and trained them
on the MNIST dataset. 

## Naive Bayes and LDA
Evaluated the performance of PCA for feature selection, followed by a Naive Bayes classifier, implemented by hand. Compared this
with the performance of the sklearn implementation of the LDA classifier. Evaluation was done on the MNIST dataset.
Run as: python naive_bayes.py train.csv

## Clustering
Implemented by hand two different clustering algorithms: K-means and Expectation-Maximization for Gaussian Mixture Models.
The data set wine.csv is the Wine data set from the UCI Machine Learning Repository.
Run as: python clustering.py wine.csv

## Probabilistic Matrix Factorization
Implements the Probabilistic Matrix Factorization algorithm, with application to recommender systems. The data set ratings.csv
contains, on each line, a tuple (user, product, rating). The output file ratings_predicted.csv contains a matrix of predicted
ratings, for all users and products.
Run as: python pmf.py ratings.csv