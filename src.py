
import pandas as pd
import nltk
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("sms-spam-collection-dataset/spam.csv", encoding='latin-1')
data_labels = data


data = data[['v1', 'v2']]


i=0
documents = []
while i < int(data.shape[0]):
    documents.append((data.iloc[i][0], data.iloc[i][1].split()))
    i+=1


#shuffling as to not get bias data
random.shuffle(documents)


#Function that turns spam-ham dataset to 0-1 table

labels = data_labels[['v1']]
span_ham_list = []
for i in labels.values:
    span_ham_list.extend(i)

def spam_ham_converter(lst):
    new_a = np.array([])
    for i in lst:
        if i == 'ham':
            new_a = np.concatenate((new_a, np.array([0])))
            continue
        new_a = np.concatenate((new_a, np.array([1])))
    return new_a.reshape(5572,1)

spam_ham_list = spam_ham_converter(span_ham_list)
Y = spam_ham_list
Y

every_word = []
for j in data["v2"]:
    j = j.lower()
    split_word= j.split()
    every_word.extend(split_word)


#Frequency distribution in NLTK to remove useless words
every_word=nltk.FreqDist(every_word)


word_features = [x[0] for x in every_word.most_common(1500)]


def find_features(d):
    array = np.array([w in d for w in word_features])
    return array


features_set = np.array([])
for m in data["v2"]:
    features_set = np.append(features_set, find_features(m))

features_set = features_set.reshape(5572, 1500)


##TRAINING TIME

##Separate Data
X_train, X_test, Y_train, Y_test = train_test_split(features_set,Y,test_size=0.10)

lg = LogisticRegression()
lg.fit(X_train, Y_train)

def error(actual_y, predict_y):
    actual_y = actual_y.reshape(predict_y.shape[0],)
    counter = 0
    for i in range(predict_y.shape[0]):
        if actual_y[i] != predict_y[i]:
            counter += 1
    return counter/predict_y.shape[0]

# Fit your model on the training set
Y_fitted = lg.predict(X_train)

# Predict housing prices on the test set
Y_pred = lg.predict(X_test)


train_error = error(Y_train, Y_fitted)

test_error = error(Y_test, Y_pred)

print("Accuracy of model for testing data: " + str(1-test_error))
print("Accuracy of model for training data: " + str(1-train_error))


plt.hist([Y_fitted, Y_train.reshape(Y_fitted.shape[0],)])


plt.hist([Y_pred, Y_test.reshape(Y_pred.shape[0],)])
