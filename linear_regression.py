#import library
import pandas as pd
import matplotlib.pyplot as plt

#import data
df=pd.read_csv("original.csv",sep=";")
print(df)

#plot data
# plt.scatter(df.deneyim,df.maas)
# plt.xlabel("deneyim")
# plt.ylabel("maas")
#plt.show()

#%% Linear regression

#sklearn library
from sklearn.linear_model import LinearRegression
#linear regression model
linear_reg=LinearRegression()
# x ve y değerleri pandas type.
# Fakat sklearn numpy ile verimli .values eklememiz gerek
#bir başka konu ise df.deneyim.values shape (14,) görünür
#ancak sklearn bunu 14,1 olarak görmek ister bu yüzden
#reshape(-1,1) kullanmamız gerek
#Buraya kadar olan tüm bölüm grafikte mavi bölgedir
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

#Şimdi x ve y eksenini bir birinden ayırıp fit edeceğiz
linear_reg.fit(x,y)

#%% predection
import numpy as np
#line ın y eksenini kesitiği nokta bulma işlemi görsel b0
b0=linear_reg.predict([[0]])
print("b0:",b0)
b0_=linear_reg.intercept_ #line y eksenini kestiği noktayı test etmek için
print("b0_",b0_) # y eksenini kesitiği nokta inercept

b1=linear_reg.coef_
print("b1:",b1) #eğim slope

#maas=1663+1138.deneyim

#örneğin 11 yıllık deneyim için maaş hesaplayalım
maas_yeni=1663+1138*11 #bu manuel yöntem
print("Manuel Yöntem:",maas_yeni)
print(linear_reg.predict([[11]]))

#visualize line
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(-1,1) #deneyim (16,) olmaması için reshape koymamız gerek
plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(array) #maas
plt.plot(array, y_head, color='red')
