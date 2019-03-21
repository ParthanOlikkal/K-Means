import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import maplotlib.pyplot as plt
%matplotlib inline

#Load Dataset
url = "https://raw.githubusercontent.com/ParthanOlikkal/K-Means/master/Cust_Segmentation.csv"
cust_df = pd.read_csv(url)
cust_df.head()

#Dropping Address field as it is a categorical value
df = cust_df.drop('Address', axis=1)
df.head()

#Normalizing the dataset
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clust_dataSet = StandardScaler().fit_transform(X)
Clust_dataSet

#Modeling
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

#Insights
df["Clus_km"] = labels
df.head(5)

#Centroid values are checked by averaging the features in each cluster
df.groupby('Clus_km').mean()

#Looking at the distribution of customers based on their age and income
area = np.pi * (X[:,1])**2
plt.scatter(X[:,0], X[:,3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize = 18)
plt.ylabel('Income', fontsize = 16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize = (8,6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:,0], X[:,3], c = labels.astype(np.float))

#3 clusters : 
#	- Affluent, educated and old age
#	- Middle aged and middle income
#	- young and low income
