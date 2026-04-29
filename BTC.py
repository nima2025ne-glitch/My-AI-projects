import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense

sc = MinMaxScaler(feature_range=(0,1))
ds = pd.read_csv('BTC-USD.csv')
ds.head()
ds.shape
closed = ds[['Date','Close']]
closed.head()
plt.close()
plt.figure(1,figsize=(12,6))
plt.plot(closed['Date'],closed['Close'])
plt.show()
price = closed[closed['Date'] >= '2020-01-01']
price
x = price['Close']

x = sc.fit_transform(np.array(x).reshape(-1,1))
x
xTrain = x[0:625]
xTest = x[625:]
x1 = []
y1 = []
a = []
b = 0

for i in range(len(xTrain)-10-1):
    a = xTrain[i:i+10,0]
    b = xTrain[i+10,0]
    x1.append(a)
    y1.append(b)

xtrain = np.array(x1)
ytrain = np.array(y1)
print(xtrain.shape)

x1 = []
y1 = []
a = []
b = 0

for i in range(len(xTest)-10-1):
    a = xTest[i:i+10,0]
    b = xTest[i+10,0]
    x1.append(a)
    y1.append(b)

xtest = np.array(x1)
ytest = np.array(y1)
print(xtest.shape)
xtrain = xtrain.reshape(614,10,1)
xtest = xtest.reshape(145,10,1)
print(xtest.shape)
print(xtrain.shape)
model = Sequential()

model.add(LSTM(10,activation='relu',input_shape=(None,1)))
model.add(Dense(1))

model.compile(loss="mean_squared_error" , optimizer='adam')
h = model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=500)
train_predict=model.predict(xtrain)
test_predict=model.predict(xtest)
train_predict.shape, test_predict.shape
train_predict = sc.inverse_transform(train_predict)
test_predict = sc.inverse_transform(test_predict)
original_ytrain = sc.inverse_transform(ytrain.reshape(-1,1)) 
original_ytest = sc.inverse_transform(ytest.reshape(-1,1)) 
print("orginal : ",original_ytest[1])
print("robot :  ",test_predict[1])
plt.close()


plt.figure(1,(12,6))


plt.plot(original_ytest)

plt.plot(test_predict)

plt.show()
