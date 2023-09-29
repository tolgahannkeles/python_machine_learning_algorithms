#https://prnt.sc/YHekehqyUhee
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Software&Design/MachineLearning/News/Reinforced Learning/UCB/Ads_CTR_Optimisation.csv")

import random
#UCB


N=len(data)# 10.000 tıklama
d=len(data.columns)# toplam 10 ilan var
#Ri(
#Ni(n)
toplam = 0 # toplam odul
secilenler = []
ones = [0] * d
zeros = [0] * d

for n in range(1,N):
    ad = 0 #seçilen ilan
    max_th = 0

    for i in range(0,d):
        rasbeta = random.betavariate ( ones[i] + 1 , zeros[i] +1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad = i

    secilenler.append(ad)
    odul = 1 # verilerdeki n. satır = 1 ise odul 1z
    if data.values[n,ad] == odul:
        ones[ad]+=1
        toplam+= 1
    else :
        zeros[ad]+=1
        toplam+=0


print('Toplam Odul:')   
print(toplam)

