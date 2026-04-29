import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
import tensorflow as tf

# بارگذاری داده
ds = pd.read_csv("behdadkarimi/all.csv")

# حذف ستون‌های نامعلوم و بی‌ربط
ds = ds.drop(['replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'hashtags', 'sourceLabel'], axis=1)

# رمزگذاری برچسب‌های عاطفه
to_numeric = LabelEncoder()
ds['emotion'] = to_numeric.fit_transform(ds['emotion'])
y = ds['emotion']
x = ds['tweet']

# توکنیزه کردن متن‌ها
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)
x2 = pad_sequences(sequences, maxlen=500)

# تقسیم داده‌ها به آموزش و آزمون
x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=0.2, random_state=42)

# کدگذاری برچسب‌ها به صورت one-hot
num_classes = ds['emotion'].nunique()

# ساخت مدل
model = Sequential()
model.add(Embedding(10000, 64))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # لایه خروجی با softmax برای دسته‌بندی چندکلاسه

# کامپایل کردن مدل
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # استفاده از loss مناسب برای برچسب‌های عددی
    metrics=['accuracy']
)

# آموزش مدل
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
