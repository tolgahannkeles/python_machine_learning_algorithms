import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Stupid Humanity/Steps of AI Advanture/MachineLearning/Clustering/k_means/customers.csv")


print(data.isnull().sum())  # null sayısını verir

x=data.iloc[:,2:4].values


from sklearn.cluster import KMeans, k_means


#İdeal cluster sayısını bulma

wsccScores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++", random_state=12) # init= "random"
    kmeans.fit(x)
    wsccScores.append(kmeans.inertia_) # WCSS scorunu verir

plt.plot(range(1,11),wsccScores)
plt.show()

kmeans=KMeans(n_clusters=3, init="k-means++", random_state=12)
kmeans.fit(x)
pred=kmeans.predict(x) # 0 1 2 şeklinde sonuç listesi döndürür

plt.scatter(x[pred==0,0],x[pred==0,1],c="red")
plt.scatter(x[pred==1,0],x[pred==1,1],c="green")
plt.scatter(x[pred==2,0],x[pred==2,1],c="blue")
plt.show()