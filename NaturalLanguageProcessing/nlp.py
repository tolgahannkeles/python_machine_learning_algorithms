import pandas as pd
reviews=pd.read_csv("C:/Software&Design/MachineLearning/News/NaturalLanguageProcessing/reviews.csv",error_bad_lines=False)

#preprocessing
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

derlem=[]

for i in range(len(reviews)):
    # Noktalama işaretlerinden ayırma
    review=re.sub("[^a-zA-Z]"," ",reviews["Review"][i])
    review=review.lower() # her şeyi küçük harf yaptı
    review=review.split() # kelimeleri boşluktan böldü


    review=[ps.stem(word)for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    derlem.append(review)

# kelime sayma
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)
x=cv.fit_transform(derlem).toarray()[:704]
y=reviews.iloc[:,1:]
y.dropna(axis=0,inplace=True)


#6Machine Learning kısmı

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred=y_pred,y_true=y_test)
print(cm)