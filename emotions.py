import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
ds = pd.read_csv("s.csv")
ds.head()
toNum = TfidfVectorizer()
x = ds['comment_cleaned']
y = ds['label']
x2 = toNum.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x2, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
print("دقت مدل:", accuracy_score(ytest, ypred))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(256, activation='relu', input_dim=12002))

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

model.fit(xtrain,ytrain,epochs=24,validation_data=(xtest,ytest))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

text = input("نظر خود را وارد کنید: ")
tn = toNum.transform([text])  
answer = model.predict(tn)

if answer[0] == 1:
    print("+")
else:
    print("-")
