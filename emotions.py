import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ds = pd.read_csv("digikala_text.csv")
toNum = TfidfVectorizer()
ds.head()
x = ds['Text']
y = ds['Suggestion']
x2 = toNum.fit_transform(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x2,y,test_size=0.2,random_state=42)
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(xtrain,ytrain)
from sklearn.metrics import accuracy_score as AS
ypred = model.predict(xtest)
AS(ypred,ytest)
text = input("نظر خود را وارد کنید: ")
tn = toNum.transform([text])
answer = model.predict(tn)

if answer[0] == 1:
    print("+")
else:
    print("-")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# بارگذاری داده‌ها
ds = pd.read_csv("digikala_text.csv")

# تبدیل متن‌ها به ویژگی‌های عددی
toNum = TfidfVectorizer()
x = ds['Text']
y = ds['Suggestion']
x2 = toNum.fit_transform(x)

# تقسیم داده‌ها به آموزش و تست
xtrain, xtest, ytrain, ytest = train_test_split(x2, y, test_size=0.2, random_state=42)

# آموزش مدل
model = SVC(kernel='linear')
model.fit(xtrain, ytrain)

# ارزیابی مدل
ypred = model.predict(xtest)
print("دقت مدل:", accuracy_score(ytest, ypred))

# دریافت ورودی از کاربر
text = input("نظر خود را وارد کنید: ")
tn = toNum.transform([text])  # فقط transform، نه fit_transform
answer = model.predict(tn)

if answer[0] == 1:
    print("+")
else:
    print("-")
