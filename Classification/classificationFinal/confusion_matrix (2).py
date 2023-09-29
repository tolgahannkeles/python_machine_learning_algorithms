"""
https://prnt.sc/T0CxDUoXabYQ

accuracy=(doğru değerler toplamı)/(veri sayısı)
kavramlar:
    https://prnt.sc/I8HisUQ8au7G
    https://prnt.sc/R31EhOJji3-n
    https://prnt.sc/xWEmlprA7kpW
"""

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_true=y_test,y_pred=y_pred)