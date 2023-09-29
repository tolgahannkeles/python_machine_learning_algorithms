"""dendogramda y eksenine göre en uzun bölümün olduğu yer optümum pointtir"""

import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Clustering/k_means/customers.csv")

# print(data.isnull().sum())  # null sayısını verir

x=data.iloc[:,2:4].values

from sklearn.cluster import AgglomerativeClustering
agg_c=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
pred=agg_c.fit_predict(x)

plt.scatter(x[pred==0,0],x[pred==0,1],s=100,c="red")
plt.scatter(x[pred==1,0],x[pred==1,1],s=100,c="green")
plt.scatter(x[pred==2,0],x[pred==2,1],s=100,c="blue")
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()