# Natural Language Processng
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting = 3)

#Cleaning the texts
import re
import nltk
nltk.download('stopwords') #list of stopwords to remove them from list
from nltk.corpus import stopwords 
#stemming 
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #cleaning execpt A to Z and a to z
    review = review.lower() #lower all the letters in lowercase
    review = review.split() #splitting data in list
    ps =PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #using for loop comparing each to stopword
    review = ' '.join(review)
    corpus.append(review)


#creating the bag of word of model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #converting corpus into sparse matrix


#splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,random_state=0)#test set with size 100


#Fitting Logistic Regression to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#predicting the test set resulting  
y_pred = classifier.predict(X_test)


#making Confusing Matrixx
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_test,y_pred)

#checkout the accuracy by (total correct prediction)/test set(100)
