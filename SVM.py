# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 13:38:09 2021

@author: Alperen Canbey
"""

import pickle 


with open('svm_preprocess.pickle', 'rb') as ___:
    training_data_svm = pickle.load(___)
    
with open('svm_preprocess_balanced.pickle', 'rb') as x:
    training_data = pickle.load(x)

with open('dataset_unique_balanced.pickle', 'rb') as output_train_unique_balanced:
    training_data = pickle.load(output_train_unique_balanced)

train_labels = []
training_set = []


test_labels = []
testing_set = []
k= 0
p = 0
n=0

for i in range (0,len(training_data)):
    if (training_data[i][1] == 'Neutral') and (k <6000):
        train_labels.append('Neutral')
        k = k+1
        training_set.append(training_data[i][0])
    elif (training_data[i][1] == 'Positive') and (n <6000):
        train_labels.append('Positive')
        n=n+1
        training_set.append(training_data[i][0])
    elif (training_data[i][1] == 'Negative') and (p < 6000):
        train_labels.append('Negative')
        training_set.append(training_data[i][0])
        p = p+1
    elif (training_data[i][1] == 'Neutral') and (k <7000) and (k >= 6000):
        test_labels.append('Neutral')
        k = k+1
        testing_set.append(training_data[i][0])
    elif (training_data[i][1] == 'Positive') and (n <7000) and (n >= 6000):
        test_labels.append('Positive')
        n=n+1
        testing_set.append(training_data[i][0])
    elif (training_data[i][1] == 'Negative') and (p < 7000) and (p >= 6000):
        test_labels.append('Negative')
        testing_set.append(training_data[i][0])
        p = p+1

        
train_labels.count('neutral')
labels.count('Positive')
labels.count('Negative')



train_vectors = vectorizer.fit_transform(training_data)
test_vectors = vectorizer.transform(testing_set)




test_labels = []
testing_set = []
k= 0
p = 0
n=0

for i in range (5000,len(training_data)):
    if (training_data[i][1] == 'Neutral') and (k <100):
        test_labels.append('neutral')
        k = k+1
        testing_set.append(training_data[i][0])
    elif (training_data[i][1] == 'Positive') and (n <100):
        test_labels.append('positive')
        n=n+1
        testing_set.append(training_data[i][0])
    elif (training_data[i][1] == 'Negative') and (p < 100):
        test_labels.append('negative')
        testing_set.append(training_data[i][0])
        p = p+1

        
labels.count('neutral')
labels.count('positive')
labels.count('negative')




from sklearn.feature_extraction.text import TfidfVectorizer



# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)



train_vectors = vectorizer.fit_transform(doc[0] for doc in training_data)







training_sentences2 = training_data
classification = Classification()
classification.set_datasets(training_sentences2, split=0.8)



train_vectors = vectorizer.fit_transform(doc[0] for doc in classification.training_set)
test_vectors = vectorizer.transform(doc[0] for doc in classification.testing_set)

train_labels =[]
for (sentence, label) in classification.training_set:
    train_labels.append(label)

test_labels =[]
for (sentence, label) in classification.testing_set:
    test_labels.append(label)



import time
from sklearn import svm
from sklearn.metrics import classification_report


# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='rbf')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()

prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1




# results

print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(test_labels, prediction_linear, output_dict=True)
print('accuracy: ', report['accuracy'])
print('positive: ', report['Positive'])
print('negative: ', report['Negative'])
print('neutral: ', report['Neutral'])













