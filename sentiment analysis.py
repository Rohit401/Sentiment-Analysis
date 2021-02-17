import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
dataset=pd.read_csv(r'C:\Users\samal\Restaurant_Reviews.tsv',delimiter='\t')
dataset.head()
dataset.shape[0]
1000
dataset.Review[5]
'Now I am getting angry and I want my damn pho.'
import nltk
import re
from nltk.corpus import stopwords
sw=stopwords.words('english')
sw
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wn=WordNetLemmatizer()
dataset.shape
(1000, 2)
Corpus=[]
for i in range(dataset.shape[0]):
    review=dataset.Review[i]
    review=re.sub('[^a-zA-Z]'," ",review)
    review=review.lower()
    review=review.split()
    data=[]
    for word in review:
        if word not  in sw:
            data.append(wn.lemmatize(word))
    review=" ".join(data)
    Corpus.append(review)
    
Corpus
pd.DataFrame(x,columns=tfid.get_feature_names())
pd.DataFrame(x)
y=dataset.Liked.values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
import keras
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=x_train.shape[1],kernel_initializer='random_uniform',activation='relu',units=200))
model.add(Dense(kernel_initializer='random_uniform',activation='relu',units=200))
model.add(Dense(kernel_initializer='random_uniform',activation='sigmoid',units=1))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
​
model.summary()
_________________________________________________________________
model.fit(x_train,y_train,epochs=10,batch_size=32)
y_pred=model.predict_classes(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
