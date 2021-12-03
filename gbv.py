import tensorflow as tf
import numpy as np
import pandas as pd
import re
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.utils.multiclass import unique_labels
from zipfile import ZipFile

with ZipFile("gbv.zip","r") as zip:
    zip.extractall()

df = pd.read_csv("Train.csv")
print(df["tweet"])
print(unique_labels(df["type"]))

def clean_text(x):
    x = x.lower()
    x = x.encode("ascii","ignore").decode()
    x = re.sub("https*\S+"," ",x)
    x = re.sub("@\S+"," ",x)
    x = re.sub("#\S+"," ",x)
    x = re.sub("\'\w+","",x)
    x = re.sub("[%s]" % re.escape(string.punctuation)," ",x)
    x = re.sub("\w*\d+\w*","",x)
    x = re.sub("\s{2,}"," ",x)
    return x

temp = []
data_to_list = df["tweet"]

for i in range(len(data_to_list)):
    temp.append(clean_text(data_to_list[i]))

def tokenize(y):
    for x in y:
        yield(word_tokenize(str(x)))

data_words = list(tokenize(temp))

def detokenize(txt):
    return TreebankWordDetokenizer().detokenize(txt)

final_data = []
for i in range(len(data_words)):
    final_data.append(detokenize(data_words[i]))

print(final_data[:5])
final_data = np.array(final_data)

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

max_words = 16000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(final_data)
sequences = tokenizer.texts_to_sequences(final_data)
tweets = pad_sequences(sequences,maxlen=max_len)
with open("tokenizer.pickle","wb") as handle:
	pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
print(tweets)


dict = {"Harmful_Traditional_practice":0,"Physical_violence":1,
        "economic_violence":2,"emotional_violence":3,
        "sexual_violence":4}
df["labels"] = ""
df["labels"] = df["type"].map(dict)
labels = df["labels"]

x_train,x_test,y_train,y_test = train_test_split(tweets,labels,random_state=42)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, GRU, Dense

def model(y):
    x = Embedding(max_words,128)(y)
    x = GRU(64,return_sequences=True)(x)
    x = GRU(64)(x)
    outputs = Dense(5,activation="softmax")(x)
    model = Model(y,outputs)
    return model

model = model(Input(shape=(None,),dtype="int32"))
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

if __name__=="__main__":
    model.compile(Adam(),SparseCategoricalCrossentropy(),metrics=["accuracy"])
    checkpoint = ModelCheckpoint("gbv.h5",monitor="val_accuracy",save_best_only=True,save_weights_only=False)
    model.fit(x_train,y_train,batch_size=32,epochs=4,validation_data=(x_val,y_val),callbacks=[checkpoint])
    best = load_model("gbv.h5")
    loss,acc = best.evaluate(x_test,y_test,verbose=2)
    print("\nTest acc: {:.2f} %".format(100*acc))
    print("Test loss: {:.2f} %".format(100*loss))
