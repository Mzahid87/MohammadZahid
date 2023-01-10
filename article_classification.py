"""  
Assesment 2
As a machine learning engineer is tasked to categorize unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics.
Data can be obtained from
https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv by simply passing this URL into pd.read_csv(URL)

The following are the criteria of your project:
1) Develop your own model using LSTM which can achieve accuracy of more than 70% and F1 score of more than 0.7.
2) You are only allowed to use TensorFlow library to develop and train the model.
3) Plot the graph using Tensorboard.
4) Save the model in .h5 format in a folder named saved_models.
5) Save tokenizer in .json format in a folder named saved_models
Hint: you may train and display tensorboard using Google Colab then download the trained model after training and screenshot respectively.

"""
#%%
#1. Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, datetime
import json
import re
import joblib
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from modules import text_cleaning, lstm_model

#%%
# 2. URL Data Loading

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df_article = pd.read_csv(URL)

#%%
# 3. EDA / Data inspection
df_article.describe()
df_article.info()
df_article.head()
#%%
df_article.tail()

#%%
# 3.1 To identify duplicate article by category -->99 duplicate 
df_article.duplicated().sum()
#df_article.isna().sum()

#%%
# 3.2 To check NaN --> 0 NaN for category and text
df_article.isna().sum()

#%%
# 3.3 To print selected article for model selection
print(df_article['text'][1])

#%%
# 4. Data cleaning
#    1) Remove small letter and change to capital
#   1) Remove spacing
# 2) Remove number
# 2) Remove HTML tags
# 3) Remove punctuation
# 4) Change all to lowercase

for index, data in enumerate(df_article['text']):
    df_article['text'][index] = re.sub('<.*?', '',data)
    df_article['text'][index] = re.sub('[^a-zA-Z]', ' ', df_article['text'][index]).lower()

df_article = df_article.drop_duplicates()

#%%
# 5. Future selection
X = df_article['text']
y = df_article['category']

#%%
# 6. Data Processing
#Tokenizer
num_words = 5000 #unique number of words in all sentences
oov_token = '<OOV>' #out of vocab

# 6.1 Instantiate tokenizer
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token) #instantiate
tokenizer.fit_on_texts(X)
#word_index = tokenizer.word_index
#print(dict(list(word_index.items())[0:10]))

# 6.2 To transform the text using tokenizer --> mms.transform
X = tokenizer.texts_to_sequences(X)

# 6.3 Padding
X = pad_sequences(
    X, maxlen=200, padding='post', truncating='post')

#%%
# 6.4 One hot encoder
# to instantiate
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y[::, None])

# 6.5 Train Test Split
#expand dimension before feeding to train_test_split
#padded_text = np.expand_dims(padded_text, axis=-1 )
X_train,X_test,y_train,y_test = train_test_split(X, y, shuffle=True,test_size=0.25, random_state=123)

#%%
# 7. Model development
embedding_layer = 64

model = Sequential()
model.add(Embedding(num_words,embedding_layer)) #modify new layers
model.add(LSTM(embedding_layer, return_sequences=True)) #add embedding layers
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2,activation='softmax'))
model.summary()

# 7.1 create callback function
#early stopping callback
es = keras.callbacks.EarlyStopping(patience=10, verbose=0, restore_best_weights=True)

# 7.2 Tensnsorboard callback
log_dir='log_dir'
tensorboard_log_dir = os.path.join(log_dir, 'overfit_demo', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = keras.callbacks.TensorBoard(tensorboard_log_dir)

# %%
# 8. Model Development

model = lstm_model(num_words, y.shape[1])
model.fit(X,y, epochs = 5, batch_size = 64, validation_data = (X_test, y_test), callbacks = [tb, es])

# %%
# 9. Model Analysis
y_predicted =model.predict(X_test)

#%%
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis = 1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

#%%
disp= ConfusionMatrixDisplay(cm)
disp.plot()

# %%
# 10. Save trained tf model
#Save the model in .h5 format in a folder named saved_models.
model.save('saved_models/model.h5')

# 10.1 encoder ohe
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

#%%
# 10.2 tokenizer
token_json = tokenizer.to_json()
with open('tokenizer.json','w') as f:
    json.dump(token_json, f)

# %%
# 10.3 Save tokenizer in .json format in a folder named saved_models
joblib.dump(tokenizer, 'saved_models/tokenizer.json')

#%%
