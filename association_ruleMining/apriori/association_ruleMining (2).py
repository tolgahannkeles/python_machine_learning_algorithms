"""
breadth first search
support(a)= number(a)/number(all)
confidence(a->b)= number(a AND b)/number(a)
lift(a->b)= confidence(a->b)/support(b)
"""

"""
bir kütühane bul bu amk şeyine
"""
import pandas as pd
data=pd.read_csv("C:/Software&Design/MachineLearning/News/association_ruleMining/apriori/sepet.csv",header=None)


new=[]

for i in range(0,7501):
    new.append([str(data.values[i,j])for j in range(0,20)])



from apyori import apriori

rules=apriori(new,min_support=0.01,min_confidence=0.2, min_lenght=2,min_lift = 3) #
print(list(rules))


