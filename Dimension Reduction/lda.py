"""Linear discriminant analysis
pcaeye ek olarak sınıflar arasını maximize etmeye çalışır

"""

# LDA sınıfları gözeterek indirgeme yapar

import pandas as pd

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Dimension Reduction/wine.csv")

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#------- PCA -------
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

#------- LDA -------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)


from sklearn.linear_model import LogisticRegression

# PCA'siz ve LDA'siz
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

# PCA'li
lr_pca=LogisticRegression()
lr_pca.fit(X_train_pca,y_train)
y_pred_pca=lr_pca.predict(X_test_pca)

#LDA
lr_lda=LogisticRegression()
lr_lda.fit(X_train_lda,y_train)
y_pred_lda=lr_lda.predict(X_test_lda)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true=y_test,y_pred=y_pred)
print(f"True - PCA'siz \n{cm}")

cm=confusion_matrix(y_true=y_test,y_pred=y_pred_pca)
print(f"True - PCA'li \n{cm}")

cm=confusion_matrix(y_true=y_test,y_pred=y_pred_lda)
print(f"True - LDA'li \n{cm}")


