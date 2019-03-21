import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
form sklearn.datasets.sample_generator import make_blobs
%matplotlib_inline

#Creating random generated dataset
#Using numpy's random.seed()
np.random.seed(0)

#Make random clusters of points by using the make_blobs class
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,-3], [1,1], cluster_std=0.9])
#Here output is X = feature matrix, y = response vector
      input is n_samples, centers, cluster_std

#Scatter Plot 
plt.scatter(X[:, 0], X[:,1], marker='.')

#KMeans class using 3 parameters : init-to initialize method of the centroid
				   n_clusters-number of clusters to form 
				   n_init-number of times the kmeans algo will run

k_means = KMeans(init = "k-means++", n_clusters=4, n_init=12)

#Fitting X in Kmeans
k_means.fit(X)

#Take labels n store it in another variable
k_means_labels = k_means.labels_
k_means_labels

#Also the centers of the clusters are taken
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#Creating a Visual Plot

#Initialize the plot with specified dimensions
fig = plt.figure(figsize=(6,4))

#Colors uses a colormap, which will produce an array of colors based on the number of labels there are. We use ser(k_means_labels) to get the unique labels.
colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))

#Create a plot
ax = fig.add_subplot(1,1,1)

#For loop that plots the data points and centroids k will range from 0-3, which will match the possible clusters that each data point is in
for k,col in zip(range(len([[4,4], [-2,-1], [2,-3], [1,1]])), colors):
	#Create a list of all data points, where the data points that are in the cluster are labeled as true, else they are labeled as false
	my_members = (k_means_labels == k)

	#Define the centroid, or cluster center
	cluster_center = k_means_cluster_centers[k]

	#Plots the datapoints with color col.
	ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

	#Plots the centroids with sepcific color, but with darker outline
	ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredge='k', markersize=6)

#Title of the plot
ax.set_title('Kmeans')

#Remove x-axis ticks
ax.set_xticks(())

#Remobe y-axis ticks
ax.set_yticks(())

#Show the plot 
plt.show()